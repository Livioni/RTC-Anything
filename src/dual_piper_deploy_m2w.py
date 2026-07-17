#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import signal
import numpy as np
import argparse
from collections import deque, defaultdict
import threading

from pyAgxArm import create_agx_arm_config, AgxArmFactory

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from utils.policy_client import PolicyClient
from utils.real_time_chunking import RealTimeChunkingBuffer
from utils.wrist_history import sample_timestamped_history

import h5py
import os
from pathlib import Path
import termios
import select
import tty
import yaml

def get_bootstrap_config_path():
    """Read --config from sys.argv before argparse builds the full CLI."""
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--config" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def load_config(config_path=None):
    """Load deployment config from CLI, environment, or common project locations."""
    config_candidates = []
    if config_path:
        explicit_config_path = Path(config_path)
        if not explicit_config_path.exists():
            raise FileNotFoundError(f"Explicit --config path does not exist: {explicit_config_path}")
        config_candidates.append(explicit_config_path)
    if os.getenv("RTC_CONFIG"):
        config_candidates.append(Path(os.environ["RTC_CONFIG"]))
    project_root = Path(__file__).resolve().parents[1]
    config_candidates.extend([
        Path.cwd() / "config.yaml",
        project_root / "configs" / "config.yaml",
        project_root / "configs" / "example_config.yaml",
    ])

    for config_path in config_candidates:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            print(f"[config] Loaded dual-piper config from {config_path}")
            return loaded_config

    raise FileNotFoundError(
        "No config file found. Use --config, set RTC_CONFIG, or create configs/config.yaml."
    )


config = load_config(get_bootstrap_config_path())

# Rollout counters.
episode_idx_dict = config.get("episode_idx_dict", {"success": 0, "fail": 0})
outcome_count_dict = {"success": 0, "fail": 0}
success_count = 0
total_count = 0

# Asynchronous inference state.
_action_prod_thread = None
_action_stop_event = threading.Event()
_ACTION_DIM = 14

# Stop signal shared with the action producer thread.
stop_signal = threading.Event() 

# Dual-Piper action/state layout: dims 0..6 are left, dims 7..13 are right.
# [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
_LEFT_JOINT_SLICE = slice(0, 6)
_LEFT_GRIPPER_INDEX = 6
_RIGHT_JOINT_SLICE = slice(7, 13)
_RIGHT_GRIPPER_INDEX = 13

# HDF5 rollout saving.
def save_data(images, actions, states, success, folder_type, args):
    """
    Save rollout data to an HDF5 file.
    """
    print("final长度：",len(images), len(actions), len(states))
    global episode_idx_dict 
    data_size = len(actions)
    if data_size == 0:
        print("没有数据可保存(There is no data to save)")
        return
    
    if len(images) == 0:
        print("没有图像可保存(There are no images to save)")
        return

    image_names = list(images[0].keys())
    if not image_names:
        print("没有启用的相机可保存(No enabled cameras to save)")
        return

    # Initialize the LeRobot-compatible data structure.
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [], # fill with zeros
        '/observations/effort': [], # fill with zeros
        '/action': [],
        '/base_action': [], # fill with zeros
        '/isSuccess': []
    }
    for image_name in image_names:
        data_dict[f'/observations/images/{image_name}'] = []
    
    bridge = CvBridge()
    
    # Align images, actions, and states by rollout step.
    for i, action in enumerate(actions):
        if i < len(images):
            image_dict = images[i]
            for image_name in image_names:
                data_dict[f'/observations/images/{image_name}'].append(
                    bridge.imgmsg_to_cv2(image_dict[image_name], 'passthrough')
                )
        
        if i < len(states):
            data_dict['/observations/qpos'].append(states[i])
            # qvel/effort fill with zeros (pyAgxArm does not support reading this information))
            data_dict['/observations/qvel'].append(np.zeros(14))
            data_dict['/observations/effort'].append(np.zeros(14))

        data_dict['/action'].append(action)
        data_dict['/isSuccess'].append(success)
        data_dict['/base_action'].append([0.0, 0.0])
    
    print("data_dict['/observations/qpos']:" , len(data_dict['/observations/qpos']))

    if folder_type not in episode_idx_dict:
        episode_idx_dict[folder_type] = 0
    idx = episode_idx_dict[folder_type]
    base_path = os.path.join(args.output_dir, folder_type)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    dataset_path = os.path.join(base_path, f"episode_{idx}.hdf5")
    
    # Write aligned rollout arrays to HDF5.
    t0 = time.time()
    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        
        obs = root.create_group('observations')
        image = obs.create_group('images')
        
        # Infer dataset shape from the actual camera frame size.
        first_image = data_dict[f'/observations/images/{image_names[0]}'][0]
        h, w, c = first_image.shape
        
        for image_name in image_names:
            _ = image.create_dataset(image_name, (data_size, h, w, c), dtype='uint8', chunks=(1, h, w, c))
        
        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('isSuccess', (data_size))

        for name, array in data_dict.items():
            root[name][...] = array
    
    print(f'\033[32m Saving: {time.time() - t0:.1f} secs. {dataset_path} \033[0m')
    episode_idx_dict[folder_type] += 1
# ROS image synchronization.
class RosOperator:
    def __init__(self, args):
        self.img_front_deque = None
        self.img_front = None
        self.img_left = None
        self.img_right = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._last_frame_wait_log = 0.0
        self._last_sync_wait_log = 0.0
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        raw_history_frames = max(1, int(config.get('wrist_raw_history_frames', 16)))
        self.left_wrist_camera_history = deque(maxlen=raw_history_frames)
        self.right_wrist_camera_history = deque(maxlen=raw_history_frames)
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()

    @staticmethod
    def _image_timestamp(msg):
        return float(msg.header.stamp.to_sec())

    def configure_wrist_history(self, raw_history_frames):
        """Resize the callback-driven wrist history rings without losing recent frames."""
        raw_history_frames = max(1, int(raw_history_frames))
        with self._image_lock:
            self.left_wrist_camera_history = deque(
                list(self.left_wrist_camera_history)[-raw_history_frames:],
                maxlen=raw_history_frames,
            )
            self.right_wrist_camera_history = deque(
                list(self.right_wrist_camera_history)[-raw_history_frames:],
                maxlen=raw_history_frames,
            )

    def start_episode_camera_timeline(self):
        """Discard pre-episode frames and wait for fresh camera callbacks."""
        with self._image_lock:
            self.img_front_deque.clear()
            self.img_left_deque.clear()
            self.img_right_deque.clear()
            self.left_wrist_camera_history.clear()
            self.right_wrist_camera_history.clear()
            self.img_front = None
            self.img_left = None
            self.img_right = None
    
    def get_img(self):
        
        if len(self.img_front_deque) == 0 :
            return False
        frame_time = self.img_front_deque[-1].header.stamp.to_sec()

        if len(self.img_front_deque) == 0 < frame_time:
            return False
        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')
        return img_front
    def get_frame(
        self,
        wrist_history_frames=0,
        wrist_temporal_stride=1,
        wrist_camera_fps=30.0,
        camera_sync_tolerance_ms=40.0,
    ):
        """Return a synchronized frame and wrist clips sampled from camera callbacks."""
        wrist_history_frames = max(0, int(wrist_history_frames))
        camera_sync_tolerance_ms = float(camera_sync_tolerance_ms)
        if (
            not np.isfinite(camera_sync_tolerance_ms)
            or camera_sync_tolerance_ms < 0
        ):
            raise ValueError("camera_sync_tolerance_ms must be finite and non-negative.")
        with self._image_lock:
            active_deques = []
            active_names = []
            if self.args.use_front:
                active_deques.append(self.img_front_deque)
                active_names.append(("front", self.args.img_front_topic, self.img_front_deque))
            if self.args.use_wrist and self.args.use_left_wrist:
                active_deques.append(self.img_left_deque)
                active_names.append(("left_wrist", self.args.img_left_topic, self.img_left_deque))
            if self.args.use_wrist and self.args.use_right_wrist:
                active_deques.append(self.img_right_deque)
                active_names.append(("right_wrist", self.args.img_right_topic, self.img_right_deque))
            if not active_deques:
                raise ValueError("At least one camera must be enabled for dual-arm deployment.")
            if any(len(image_deque) == 0 for image_deque in active_deques):
                self._log_frame_wait(active_names)
                return False

            frame_time = min(
                self._image_timestamp(image_deque[-1])
                for image_deque in active_deques
            )

            selected = {}

            def select_nearest(name, image_deque):
                index = min(
                    range(len(image_deque)),
                    key=lambda item_index: abs(
                        self._image_timestamp(image_deque[item_index]) - frame_time
                    ),
                )
                selected[name] = (image_deque, index, image_deque[index])

            if self.args.use_front:
                select_nearest("front", self.img_front_deque)
            if self.args.use_wrist and self.args.use_left_wrist:
                select_nearest("left_wrist", self.img_left_deque)
            if self.args.use_wrist and self.args.use_right_wrist:
                select_nearest("right_wrist", self.img_right_deque)

            selected_timestamps = {
                name: self._image_timestamp(msg)
                for name, (_image_deque, _index, msg) in selected.items()
            }
            timestamp_span_ms = (
                max(selected_timestamps.values())
                - min(selected_timestamps.values())
            ) * 1000.0
            if timestamp_span_ms > camera_sync_tolerance_ms:
                self._log_sync_wait(
                    selected_timestamps,
                    timestamp_span_ms,
                    camera_sync_tolerance_ms,
                )
                return False

            messages = {}
            for name, (image_deque, index, _msg) in selected.items():
                for _ in range(index):
                    image_deque.popleft()
                messages[name] = image_deque.popleft()

            sampled_wrist_messages = None
            if (
                wrist_history_frames > 0
                and "left_wrist" in messages
                and "right_wrist" in messages
            ):
                left_anchor_msg = messages["left_wrist"]
                right_anchor_msg = messages["right_wrist"]
                left_anchor = (self._image_timestamp(left_anchor_msg), left_anchor_msg)
                right_anchor = (self._image_timestamp(right_anchor_msg), right_anchor_msg)
                sampled_wrist_messages = [
                    sample_timestamped_history(
                        self.left_wrist_camera_history,
                        left_anchor,
                        wrist_history_frames,
                        wrist_temporal_stride,
                        wrist_camera_fps,
                        timeline_anchor_time=frame_time,
                    ),
                    sample_timestamped_history(
                        self.right_wrist_camera_history,
                        right_anchor,
                        wrist_history_frames,
                        wrist_temporal_stride,
                        wrist_camera_fps,
                        timeline_anchor_time=frame_time,
                    ),
                ]

        converted_messages = {}

        def convert_message(msg):
            key = id(msg)
            if key not in converted_messages:
                converted_messages[key] = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            return converted_messages[key]

        frame = {
            name: convert_message(msg)
            for name, msg in messages.items()
        }
        if sampled_wrist_messages is not None:
            frame["wrist_views"] = [
                [convert_message(msg) for _timestamp, msg in clip]
                for clip in sampled_wrist_messages
            ]
        return frame

    def _log_sync_wait(self, timestamps, span_ms, tolerance_ms):
        now = time.time()
        if now - self._last_sync_wait_log < 2.0:
            return
        self._last_sync_wait_log = now
        details = ", ".join(
            f"{name}={timestamp:.6f}"
            for name, timestamp in timestamps.items()
        )
        print(
            "[camera] waiting for synchronized frames; "
            f"span={span_ms:.1f}ms > tolerance={tolerance_ms:.1f}ms; "
            f"{details}"
        )

    def _log_frame_wait(self, active_names):
        now = time.time()
        if now - self._last_frame_wait_log < 2.0:
            return
        self._last_frame_wait_log = now
        missing = [
            f"{name}({topic})"
            for name, topic, image_deque in active_names
            if len(image_deque) == 0
        ]
        counts = ", ".join(
            f"{name}={len(image_deque)}"
            for name, _topic, image_deque in active_names
        )
        print(
            "[camera] waiting for synchronized frames; "
            f"missing: {', '.join(missing) if missing else 'none'}; "
            f"queue sizes: {counts}"
        )

    def img_left_callback(self, msg):
        with self._image_lock:
            if len(self.img_left_deque) >= 2000:
                self.img_left_deque.popleft()
            self.img_left_deque.append(msg)
            self.left_wrist_camera_history.append((self._image_timestamp(msg), msg))
            self.img_left = msg

    def img_right_callback(self, msg):
        with self._image_lock:
            if len(self.img_right_deque) >= 2000:
                self.img_right_deque.popleft()
            self.img_right_deque.append(msg)
            self.right_wrist_camera_history.append((self._image_timestamp(msg), msg))
            self.img_right = msg

    def img_front_callback(self, msg):
        with self._image_lock:
            if len(self.img_front_deque) >= 2000:
                self.img_front_deque.popleft()
            self.img_front_deque.append(msg)
            self.img_front = msg

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def get_latest_image_messages(self):
        """Return latest ROS image messages keyed by configured policy camera names."""
        with self._image_lock:
            images = {}
            if self.args.use_front and self.img_front is not None:
                images[config.get("front_camera_name", "cam_high")] = self.img_front
            if self.args.use_wrist and self.args.use_left_wrist and self.img_left is not None:
                images[config.get("left_wrist_camera_name", "cam_left_wrist")] = self.img_left
            if self.args.use_wrist and self.args.use_right_wrist and self.img_right is not None:
                images[config.get("right_wrist_camera_name", "cam_right_wrist")] = self.img_right
            return images

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        if self.args.use_front:
            print(f"[ros] subscribe front: {self.args.img_front_topic}")
            rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_wrist and self.args.use_left_wrist:
            print(f"[ros] subscribe left_wrist: {self.args.img_left_topic}")
            rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_wrist and self.args.use_right_wrist:
            print(f"[ros] subscribe right_wrist: {self.args.img_right_topic}")
            rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)


# Inference and robot control.
class InferController:
    def __init__(self, host="localhost", port=8000):
        self.client = None
        self.host = host
        self.port = port
        self.left_channel = config.get('left_channel', 'can_left')
        self.right_channel = config.get('right_channel', 'can_right')
        self.bitrate = config.get('bitrate', 1000000)
        
        # Robot arm parameters.
        self.speed_pct = config.get('speed_pct', 15)
        self.max_linear_vel = config.get('max_linear_vel', 0.5)     # m/s
        self.max_angular_vel = config.get('max_angular_vel', 0.1)    # rad/s
        self.max_linear_acc = config.get('max_linear_acc', 0.1)     # m/s2
        self.max_angular_acc = config.get('max_angular_acc', 0.05)   # rad/s2

        # Gripper handles.
        self.left_gripper = None
        self.right_gripper = None
        
        # Task instruction passed to the policy.
        self.instruction = config.get('instruction', 'fold the cloth')
        
        # Execution parameters.
        self.ACTION_CHUNK_SIZE = config.get('action_chunk_size', 50)
        self.WRIST_HISTORY_FRAMES = int(config.get('wrist_history_frames', 8))
        self.WRIST_TEMPORAL_STRIDE = int(config.get('wrist_temporal_stride', 2))
        self.WRIST_CAMERA_FPS = float(config.get('wrist_camera_fps', 30.0))
        self.CAMERA_SYNC_TOLERANCE_MS = float(
            config.get('camera_sync_tolerance_ms', 40.0)
        )
        self.WRIST_RAW_HISTORY_FRAMES = int(
            config.get(
                'wrist_raw_history_frames',
                max(
                    self.WRIST_HISTORY_FRAMES * self.WRIST_TEMPORAL_STRIDE,
                    (self.WRIST_HISTORY_FRAMES - 1) * self.WRIST_TEMPORAL_STRIDE + 1,
                ),
            )
        )
        self.send_wrist_history = self._config_bool(
            config.get('send_wrist_history', self.WRIST_HISTORY_FRAMES > 1),
            default=self.WRIST_HISTORY_FRAMES > 1,
        )
        self.rtc = RealTimeChunkingBuffer(
            chunk_size=self.ACTION_CHUNK_SIZE,
            exp_weight_factor=config.get('exp_weight_factor', 0.3),
            debug=config.get('rtc_debug', False),
        )
        self.ACTION_DEBUG = self._config_bool(config.get('action_debug', False), default=False)
        if not np.isfinite(self.WRIST_CAMERA_FPS) or self.WRIST_CAMERA_FPS <= 0:
            raise ValueError("wrist_camera_fps must be a positive finite number.")
        if (
            not np.isfinite(self.CAMERA_SYNC_TOLERANCE_MS)
            or self.CAMERA_SYNC_TOLERANCE_MS < 0
        ):
            raise ValueError("camera_sync_tolerance_ms must be finite and non-negative.")
        minimum_raw_history = (
            max(1, self.WRIST_HISTORY_FRAMES) - 1
        ) * max(1, self.WRIST_TEMPORAL_STRIDE) + 1
        self.WRIST_RAW_HISTORY_FRAMES = max(
            self.WRIST_RAW_HISTORY_FRAMES,
            minimum_raw_history,
        )
        
        # Initial dual-arm pose.
        # folding_cloth
        self.LEFT_INIT_POSITION = config.get('left_init_position', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
        self.RIGHT_INIT_POSITION = config.get('right_init_position', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05])     
        
        # Safety thresholds.
        self.ACTION_SAFETY_THRESHOLD = config.get('action_safety_threshold', 1.5)
        self.STATE_SAFETY_THRESHOLD = config.get('state_safety_threshold', 0.3)

    @staticmethod
    def _config_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _rebuild_rtc_buffer(self):
        self.rtc = RealTimeChunkingBuffer(
            chunk_size=self.ACTION_CHUNK_SIZE,
            exp_weight_factor=config.get('exp_weight_factor', 0.3),
            debug=config.get('rtc_debug', False),
        )

    def _apply_server_metadata(self):
        if self.client is None:
            return
        try:
            metadata = self.client.get_server_metadata() or {}
        except Exception as e:
            print(f"[policy] failed to read server metadata: {e}")
            return
        if metadata:
            print(f"[policy] server metadata: {metadata}")

        server_chunk_size = metadata.get("action_chunk_size")
        if server_chunk_size is not None:
            server_chunk_size = int(server_chunk_size)
            if server_chunk_size > 0 and server_chunk_size != self.ACTION_CHUNK_SIZE:
                print(
                    f"[policy] action_chunk_size {self.ACTION_CHUNK_SIZE} -> "
                    f"{server_chunk_size} from server metadata"
                )
                self.ACTION_CHUNK_SIZE = server_chunk_size
                self._rebuild_rtc_buffer()

        server_history_frames = metadata.get("wrist_history_frames")
        if server_history_frames is not None:
            server_history_frames = int(server_history_frames)
            if server_history_frames > 0 and server_history_frames != self.WRIST_HISTORY_FRAMES:
                print(
                    f"[policy] wrist_history_frames {self.WRIST_HISTORY_FRAMES} -> "
                    f"{server_history_frames} from server metadata"
                )
                self.WRIST_HISTORY_FRAMES = server_history_frames
                self.send_wrist_history = self.WRIST_HISTORY_FRAMES > 1

        server_stride = metadata.get("wrist_temporal_stride")
        if server_stride is not None:
            server_stride = int(server_stride)
            if server_stride > 0 and server_stride != self.WRIST_TEMPORAL_STRIDE:
                print(
                    f"[policy] wrist_temporal_stride {self.WRIST_TEMPORAL_STRIDE} -> "
                    f"{server_stride} from server metadata"
                )
                self.WRIST_TEMPORAL_STRIDE = server_stride

        server_raw_history = metadata.get("wrist_raw_history_frames")
        default_raw_history = max(
            self.WRIST_HISTORY_FRAMES * self.WRIST_TEMPORAL_STRIDE,
            (self.WRIST_HISTORY_FRAMES - 1) * self.WRIST_TEMPORAL_STRIDE + 1,
        )
        raw_history_frames = int(server_raw_history) if server_raw_history is not None else default_raw_history
        minimum_raw_history = (
            max(1, self.WRIST_HISTORY_FRAMES) - 1
        ) * max(1, self.WRIST_TEMPORAL_STRIDE) + 1
        raw_history_frames = max(1, raw_history_frames, minimum_raw_history)
        if raw_history_frames != self.WRIST_RAW_HISTORY_FRAMES:
            print(
                f"[policy] wrist_raw_history_frames {self.WRIST_RAW_HISTORY_FRAMES} -> "
                f"{raw_history_frames} from server metadata"
            )
            self.WRIST_RAW_HISTORY_FRAMES = raw_history_frames

    def _log_action_debug(self, label, actions):
        if not self.ACTION_DEBUG:
            return
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] <= _RIGHT_GRIPPER_INDEX:
                print(f"[action_debug] {label}: invalid action shape {arr.shape}")
                return
            left_raw = float(arr[_LEFT_GRIPPER_INDEX])
            right_raw = float(arr[_RIGHT_GRIPPER_INDEX])
            print(
                f"[action_debug] {label}: shape={arr.shape} "
                f"left_gripper={left_raw:.6f}->{np.clip(left_raw, 0.0, 0.1):.6f} "
                f"right_gripper={right_raw:.6f}->{np.clip(right_raw, 0.0, 0.1):.6f}"
            )
            return
        if arr.ndim == 2:
            if arr.shape[1] <= _RIGHT_GRIPPER_INDEX:
                print(f"[action_debug] {label}: invalid action chunk shape {arr.shape}")
                return
            grippers = arr[:, [_LEFT_GRIPPER_INDEX, _RIGHT_GRIPPER_INDEX]]
            print(
                f"[action_debug] {label}: shape={arr.shape} "
                f"first={grippers[0].tolist()} "
                f"min={grippers.min(axis=0).tolist()} "
                f"max={grippers.max(axis=0).tolist()}"
            )
            return
        print(f"[action_debug] {label}: unsupported action shape {arr.shape}")

    def connect_arms(self):
        """
        Connect both Piper arms and initialize grippers.
        """
        print("连接双臂(connect arms)...")
        try:
            # Left arm.
            cfg_l = create_agx_arm_config(
                robot="piper", comm="can", channel=self.left_channel, bitrate=self.bitrate
            )
            self.left_arm = AgxArmFactory.create_arm(cfg_l)
            self.left_gripper = self.left_arm.init_effector(
                self.left_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.left_arm.connect()
            
            # Right arm.
            cfg_r = create_agx_arm_config(
                robot="piper", comm="can", channel=self.right_channel, bitrate=self.bitrate
            )
            self.right_arm = AgxArmFactory.create_arm(cfg_r)
            self.right_gripper = self.right_arm.init_effector(
                self.right_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.right_arm.connect()
            
            time.sleep(0.5)
            if not (self.left_arm.is_ok() and self.right_arm.is_ok()):
                raise Exception("机械臂连接状态检查失败(connection status check failed)")
            
            # Apply motion limits and enable both arms.
            for name, arm in [("左臂left arm", self.left_arm), ("右臂right arm", self.right_arm)]:
                arm.set_flange_vel_acc_limits(
                    max_linear_vel=self.max_linear_vel,
                    max_angular_vel=self.max_angular_vel,
                    max_linear_acc=self.max_linear_acc,
                    max_angular_acc=self.max_angular_acc,
                    timeout=1.0
                )
                arm.set_speed_percent(self.speed_pct)
                
                enabled = False
                for _ in range(5):
                    if arm.enable():
                        enabled = True
                        break
                    time.sleep(0.5)
                if not enabled:
                    raise Exception(f"{name} 使能超时(enable timeout)")
                
            print("连接成功(connect successfully)\n")
            return True
            
        except Exception as e:
            print(f"连接错误：{e}")
            return False
    
    def get_status_and_state(self):
        """
        Return a 14D state: dims 0..6 left, dims 7..13 right.
        """
        state = np.zeros(14, dtype=np.float32)
        try:
            # Left arm joints.
            ja_l = self.left_arm.get_joint_angles()
            if ja_l is not None: 
                state[_LEFT_JOINT_SLICE] = ja_l.msg
            
            # Right arm joints.
            ja_r = self.right_arm.get_joint_angles()
            if ja_r is not None: 
                state[_RIGHT_JOINT_SLICE] = ja_r.msg
            
            # Left gripper width.
            if self.left_gripper:
                gs = self.left_gripper.get_gripper_status()
                if gs is not None:
                    state[_LEFT_GRIPPER_INDEX] = gs.msg.value
            
            # Right gripper width.
            if self.right_gripper:
                gs = self.right_gripper.get_gripper_status()
                if gs is not None:
                    state[_RIGHT_GRIPPER_INDEX] = gs.msg.value
                    
        except Exception as e:
            print(f"状态读取异常(status reading error)：{e}")
        return state
        
    def move(self, position_state):
        """
        Move both arms using a 14D target: dims 0..6 left, dims 7..13 right.
        """
        
        left_arm_position = position_state[_LEFT_JOINT_SLICE].tolist()
        right_arm_position = position_state[_RIGHT_JOINT_SLICE].tolist()
        left_gripper_position = max(0.0, min(position_state[_LEFT_GRIPPER_INDEX], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[_RIGHT_GRIPPER_INDEX], 0.1))
        try:
            self.left_arm.move_js(left_arm_position)
            self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
            self.right_arm.move_js(right_arm_position)
            self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
            
            time.sleep(0.02)
        except Exception as e:
            print(f"移动失败(move failed)：{e}")
    
    def move_initial(self, position_state):
        """
        Move both arms smoothly to their initial pose.
        """
        n=50
        left_arm_position = position_state[_LEFT_JOINT_SLICE].tolist()
        right_arm_position = position_state[_RIGHT_JOINT_SLICE].tolist()
        left_gripper_position = max(0.0, min(position_state[_LEFT_GRIPPER_INDEX], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[_RIGHT_GRIPPER_INDEX], 0.1))
        current_state = self.get_status_and_state()
        left_traj = np.linspace(current_state[_LEFT_JOINT_SLICE], left_arm_position, n)
        right_traj = np.linspace(current_state[_RIGHT_JOINT_SLICE], right_arm_position, n)
        try:
           for i in range(n):
                print(left_traj[i],right_traj[i])
                self.left_arm.move_js(left_traj[i].tolist())
                self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
                self.right_arm.move_js(right_traj[i].tolist())
                self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
                time.sleep(0.02)
        except Exception as e:
            print(f"移动失败(move failed)：{e}")

    def run(self, ros_operator, args):
        global success_count, total_count, outcome_count_dict, stop_signal
        
        if not self.connect_arms(): return False
        print(self.left_arm.get_firmware())
        # Connect to the policy server.
        self.client = PolicyClient(
            self.host,
            self.port,
            image_resize_mode="stretch",
            image_size=(224, 224),
        )
        if not self.client: return False
        self._apply_server_metadata()
        ros_operator.configure_wrist_history(
            max(
                self.WRIST_RAW_HISTORY_FRAMES,
                int(config.get('wrist_camera_buffer_frames', 32)),
            )
        )
        
        while True:
            # Reset rollout buffers for this episode.
            save_states=[]
            save_actions = []
            save_images = []

            stop_signal.clear()
            _action_stop_event.clear()
            # Return to the configured initial pose.
            rospy_rate = config.get('rospy_rate', 100)
            rate = rospy.Rate(rospy_rate)
            rate.sleep()
            initial_position = np.concatenate((self.LEFT_INIT_POSITION, self.RIGHT_INIT_POSITION))
            self.move_initial(initial_position)

            # Invalidate any stale chunks from a previous producer before the operator starts.
            self.rtc.clear()
            
            # Wait for the operator to start inference.
            input("按回车键开始模型推理(Press Enter to start model inference)...")
            # Episode t=0 starts here. Subsequent history is filled by the
            # independent 30 FPS wrist-camera callbacks, not inference calls.
            ros_operator.start_episode_camera_timeline()

            # Store completed model-request latencies for this episode only.
            inference_latencies = []

            # Start the producer thread after operator confirmation.
            global _action_prod_thread
            _action_prod_thread = threading.Thread(
                target=self._action_producer_loop,
                args=(ros_operator, inference_latencies),
                daemon=True,
            )
            _action_prod_thread.start()

            # Switch stdin to non-blocking mode so space can stop the episode.
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)      
            
            t = 0
            last_action = initial_position
            st=time.time()
            stopped_by_space = False
            try:
                while not rospy.is_shutdown() and not stop_signal.is_set():
                    self.rtc.set_control_time(t)
                    # Query the real-time chunking buffer for the current action.
                    action = self.rtc.get_action(t)
                    if action is None or len(action) == 0:
                        print(f"[WAITING] Waiting for action...")
                        rate.sleep()
                        continue
                    self._log_action_debug("rtc fused action", action)
                    rate.sleep()
                    
                    # Safety check against abrupt action jumps.
                    prev_curr_l1 = np.mean(np.abs(np.array(action) - last_action))
                    if prev_curr_l1 > self.ACTION_SAFETY_THRESHOLD:
                        print(f"\033[31m 安全检查失败：动作变化过大(Safety check failed: Action change too much) {prev_curr_l1:.3f}\033[0m")
                        break
                    
                    current_state = self.get_status_and_state()
                    # Safety check against large state drift.
                    prev_state_l1 = np.mean(np.abs(current_state - action))
                    if prev_state_l1 > self.STATE_SAFETY_THRESHOLD:
                        print(f"\033[31m 安全检查失败：状态差异过大(Safety check failed: State difference too large) {prev_state_l1:.3f}\033[0m")
                        break
                    
                    last_action = np.array(action)
                    
                    # Save rollout buffers only when enabled. Camera/state are still used by the server producer.
                    if args.save_rollout:
                        image_dict = ros_operator.get_latest_image_messages()
                        if image_dict:
                            save_images.append(image_dict)
                            save_actions.append(action.copy())
                            save_states.append(current_state.copy())
                            print("保存长度：",len(save_images), len(save_actions), len(save_states))
                    # Execute the selected action.
                    print(self.rtc.get_control_time(),t)
                    print("待执行(Pending execution) action_chunk:",action )
                    try:
                        self.move(action)
                        rate.sleep()
                    except Exception as e:
                        print(f"执行异常(Execute error)：{e}")
                    
                    t += 1
                    rate.sleep()

                    # Stop the episode when the operator presses space.
                    r, w, x = select.select([sys.stdin], [], [], 0)
                    if r:
                        key = sys.stdin.read(1)
                        if key == ' ':
                            stopped_by_space = True
                            print("\n结束单次测试()(End of this episode)\n")
                            ed=time.time()
                            cost_time=ed-st
                            print(f"本次耗时(cost time): {cost_time:.2f}s, {t}steps")
                            print(t/cost_time, " steps/s")
                            break
        
            except Exception as e:
                print(f"\nerror：{e}")
            finally:
                # Restore terminal settings.
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                termios.tcflush(fd, termios.TCIFLUSH)

            # Stop the action producer thread.
            stop_signal.set() 
            _action_stop_event.set()
            if _action_prod_thread is not None and _action_prod_thread.is_alive():
                _action_prod_thread.join(timeout=2.0)

            if stopped_by_space:
                latency_snapshot = np.asarray(inference_latencies, dtype=np.float64)
                if latency_snapshot.size:
                    print(
                        f"推理延迟(Inference latency): "
                        f"平均(average) {latency_snapshot.mean() * 1000:.2f} ms, "
                        f"中位数(median) {np.median(latency_snapshot) * 1000:.2f} ms, "
                        f"样本数(samples) {latency_snapshot.size}"
                    )
                else:
                    print("推理延迟(Inference latency): 暂无成功推理样本(no successful samples)")
            
            # Ask the operator whether the episode succeeded.
            user_input = input("Success? Enter 'y' or 'n': ")
            success = True if user_input.lower() == 'y' else False
            
            if success:
                success_count += 1
                outcome_count_dict["success"] += 1
            else:
                outcome_count_dict["fail"] += 1
            total_count += 1
            
            if args.save_rollout:
                # Ask whether to save this episode.
                user_input = input("Save this episode? Enter 'y'(success/fail), or 'n'(no): ")
                save = False
                folder_type = ""
                
                if user_input.lower() == 'y':
                    save = True
                    if success==True:
                        folder_type = "success"
                    else:
                        folder_type = "fail"
                elif user_input.lower() == 'n':
                    save = False
                
                if save:
                    save_data(save_images, save_actions, save_states, success, folder_type, args)
            else:
                print("[rollout] save_rollout=false; rollout buffers and HDF5 saving are disabled.")
            self.client.reset()
            
            # Print rollout success statistics.
            try:
                success_rate = success_count / total_count * 100
            except:
                success_rate = 0.0
            
            print("\n" + "="*50)
            print("当前测试次数(present rollout episodes):", total_count)
            print("此次是否成功(success?):", "是" if success else "否")
            print("当前成功率(success rate):", f"{success_rate:.2f}%")
            print("当前成功次数(present successful episodes):", outcome_count_dict["success"])
            print("当前失败次数(present failed episodes):", outcome_count_dict["fail"])
            print("总测试次数(total episodes):", total_count)
            print("="*50 + "\n")
            
            # Ask whether to continue with another episode.
            continue_input = input("继续下一次测试？(Continue to next episode?) Enter 'y' or 'n': ")
            if continue_input.lower() != 'y':
                break
        
        self._cleanup()

    def _action_producer_loop(self, ros_operator, inference_latencies):
        """
        Capture frames, request policy actions, and cache action chunks.
        """
        rospy_rate = config.get('rospy_rate', 100)
        rate = rospy.Rate(rospy_rate)
        print_flag_local = True
        while not _action_stop_event.is_set() and not rospy.is_shutdown() and not stop_signal.is_set():
            cursor = self.rtc.get_control_time()
            generation = self.rtc.get_generation()
            already_inferred = self.rtc.has_chunk(cursor)

            if already_inferred:
                rate.sleep()
                continue
            try:
                result = ros_operator.get_frame(
                    wrist_history_frames=(
                        self.WRIST_HISTORY_FRAMES if self.send_wrist_history else 0
                    ),
                    wrist_temporal_stride=self.WRIST_TEMPORAL_STRIDE,
                    wrist_camera_fps=self.WRIST_CAMERA_FPS,
                    camera_sync_tolerance_ms=self.CAMERA_SYNC_TOLERANCE_MS,
                )
            except Exception as e:
                print(f"Camera frame error: {e}")
                rate.sleep()
                continue
            if not isinstance(result, dict) or not result:
                if print_flag_local:
                    print("async syn fail")
                    print_flag_local = False
                rate.sleep()
                continue
            print_flag_local = True

            wrist_views = result.pop("wrist_views", None)
            images = {}
            if "front" in result:
                images[config.get("front_camera_name", "cam_high")] = result["front"]
            if "left_wrist" in result:
                images[config.get("left_wrist_camera_name", "cam_left_wrist")] = result["left_wrist"]
            if "right_wrist" in result:
                images[config.get("right_wrist_camera_name", "cam_right_wrist")] = result["right_wrist"]
            # Build the model observation.
            obs={
                "images": images,
                "state": self.get_status_and_state(),
                "prompt": self.instruction
            }
            if wrist_views is not None:
                obs["wrist_views"] = wrist_views
            self.client.update_observation(obs)

            # Call the OpenPI-compatible local inference server.
            '''
            Replace this block with local model inference if you do not use the server.
            The model receives obs as input and outputs a 14-dimensional action sequence.
            '''
            try:
                inference_start = time.perf_counter()
                action_chunk = self.client.get_action()
                inference_latencies.append(time.perf_counter() - inference_start)
                self._log_action_debug("server raw chunk", action_chunk)
                action_chunk = action_chunk[:self.ACTION_CHUNK_SIZE]
                self._log_action_debug("client used chunk", action_chunk)
                if np.asarray(action_chunk).ndim != 2 or np.asarray(action_chunk).shape[-1] != _ACTION_DIM:
                    raise ValueError(
                        f"Expected action chunk shape [T, {_ACTION_DIM}], got {np.asarray(action_chunk).shape}"
                    )
                self.rtc.enqueue(action_chunk, cursor, generation=generation)
            except Exception as e:
                print(f"Inference error: {e}")
            rate.sleep()
            

    def _cleanup(self):
        """Clean up the producer thread before program exit."""
        print("\n正在退出(Exiting)...")
        _action_stop_event.set()
        
        if _action_prod_thread is not None and _action_prod_thread.is_alive():
            _action_prod_thread.join(timeout=2.0)
            print("producer线程已停止(producer thread stopped).")
        
        print("Inference stopped.")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', type=str, default=None, required=False,
                        help='Path to YAML deployment config.')
    parser.add_argument('--host', action='store', type=str, default=config.get('server_host', 'localhost'), required=False,
                        help='Model server host.')
    parser.add_argument('--port', action='store', type=int, default=config.get('server_port', 8000), required=False,
                        help='Model server port.')
    parser.add_argument('--save-rollout', dest='save_rollout', action=argparse.BooleanOptionalAction,
                        default=config.get('save_rollout', True), required=False,
                        help='Enable image/state/action buffering and HDF5 rollout saving.')
    parser.add_argument('--use-front', dest='use_front', action=argparse.BooleanOptionalAction,
                        default=config.get('use_front', True), required=False,
                        help='Enable front/high camera.')
    parser.add_argument('--use-wrist', dest='use_wrist', action=argparse.BooleanOptionalAction,
                        default=config.get('use_wrist', True), required=False,
                        help='Enable wrist cameras.')
    parser.add_argument('--use-left-wrist', dest='use_left_wrist', action=argparse.BooleanOptionalAction,
                        default=config.get('use_left_wrist', True), required=False,
                        help='Enable left wrist camera.')
    parser.add_argument('--use-right-wrist', dest='use_right_wrist', action=argparse.BooleanOptionalAction,
                        default=config.get('use_right_wrist', True), required=False,
                        help='Enable right wrist camera.')
    _topics = config.get("ros_topics_dual", {}) or {}
    parser.add_argument('--img_front_topic', action='store', type=str,
                        default=_topics.get('img_front_topic', '/camera_h/color/image_raw'), required=False)
    parser.add_argument('--img_left_topic', action='store', type=str,
                        default=_topics.get('img_left_topic', '/camera_l/color/image_raw'), required=False)
    parser.add_argument('--img_right_topic', action='store', type=str,
                        default=_topics.get('img_right_topic', '/camera_r/color/image_raw'), required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str,
                        default=_topics.get('img_front_depth_topic', '/camera_h/depth/image_raw'), required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str,
                        default=_topics.get('img_left_depth_topic', '/camera_l/depth/image_raw'), required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str,
                        default=_topics.get('img_right_depth_topic', '/camera_r/depth/image_raw'), required=False)
    parser.add_argument('--use-depth-image', '--use_depth_image', dest='use_depth_image',
                        action=argparse.BooleanOptionalAction,
                        default=config.get('use_depth_image', False), required=False)

    parser.add_argument('--output_dir', action='store', type=str, 
                        default=config.get('output_dir', './'), required=False)

    args, unknown = parser.parse_known_args()

    return args


def main():
    # Register SIGINT handling for Ctrl+C cleanup.
    def signal_handler(sig, frame):
        print("\n收到中断信号，正在清理(Receive interrupt signal, cleaning up)...")
        global stop_signal, _action_stop_event
        stop_signal.set()
        _action_stop_event.set()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    args = get_arguments()
    ros_operator = RosOperator(args)
    try:
        controller = InferController(host=args.host, port=args.port)
        controller.run(ros_operator,args)
    finally:
        pass

if __name__ == "__main__":
    main()
