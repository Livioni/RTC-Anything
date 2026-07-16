#!/usr/bin/env python3
"""Create a three-camera MP4 preview from a rollout HDF5 episode.

The default camera order is left wrist, main/high camera, right wrist.  When
the input is below ``experiments/``, the output mirrors that relative path
under ``videos/``.  For example:

    python tools/visualization.py experiments/M2W-VLA/put_mango/success/episode_0.hdf5

creates ``videos/M2W-VLA/put_mango/success/episode_0.mp4``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPOSITORY_ROOT / "experiments"
VIDEOS_ROOT = REPOSITORY_ROOT / "videos"
DEFAULT_CAMERA_KEYS = ("cam_left_wrist", "cam_high", "cam_right_wrist")
DEFAULT_CAMERA_LABELS = ("Left wrist", "Main", "Right wrist")
LABEL_HEIGHT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate left-wrist, main, and right-wrist episode views into an MP4."
    )
    parser.add_argument("episode", type=Path, help="Path to an episode .hdf5 file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output .mp4 path. By default, mirrors the path below experiments/ into videos/.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Output frame rate (default: 30).",
    )
    parser.add_argument(
        "--camera-keys",
        nargs=3,
        metavar=("LEFT", "MAIN", "RIGHT"),
        default=DEFAULT_CAMERA_KEYS,
        help="Three dataset names below observations/images, in output order.",
    )
    parser.add_argument(
        "--input-color-order",
        choices=("bgr", "rgb"),
        default="rgb",
        help="Color order of uncompressed image arrays (default: rgb).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not add camera-name labels above the three panels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output video.",
    )
    return parser.parse_args()


def default_output_path(episode_path: Path) -> Path:
    """Map experiments/<path>/episode.hdf5 to videos/<path>/episode.mp4."""
    try:
        relative_episode = episode_path.relative_to(EXPERIMENTS_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{episode_path} is not below {EXPERIMENTS_ROOT}. "
            "Pass --output to choose an explicit output path."
        ) from exc
    return VIDEOS_ROOT / relative_episode.with_suffix(".mp4")


def decode_frame(value: np.ndarray, input_color_order: str) -> np.ndarray:
    """Return one image as a BGR uint8 frame suitable for OpenCV video output.

    Rollouts written by this repository contain HxWxC uint8 frames.  The
    encoded-frame branch also supports common variable-length JPEG/PNG HDF5
    datasets, so the tool remains useful for compressed rollout files.
    """
    frame = np.asarray(value)

    if frame.ndim == 1:
        decoded = cv2.imdecode(frame.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("could not decode an encoded image frame")
        return decoded

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        if input_color_order == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        conversion = cv2.COLOR_RGBA2BGR if input_color_order == "rgb" else cv2.COLOR_BGRA2BGR
        frame = cv2.cvtColor(frame, conversion)
    else:
        raise ValueError(f"unsupported image shape {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def panel_dimensions(frame: np.ndarray, target_height: int) -> tuple[int, int]:
    height, width = frame.shape[:2]
    return max(1, round(width * target_height / height)), target_height


def resize_panel(frame: np.ndarray, dimensions: tuple[int, int]) -> np.ndarray:
    if frame.shape[1] == dimensions[0] and frame.shape[0] == dimensions[1]:
        return frame
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def add_label(panel: np.ndarray, label: str) -> np.ndarray:
    labelled = np.zeros((panel.shape[0] + LABEL_HEIGHT, panel.shape[1], 3), dtype=np.uint8)
    labelled[LABEL_HEIGHT:] = panel
    cv2.putText(labelled, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return labelled


def compose_frame(
    frames: list[np.ndarray], panel_sizes: list[tuple[int, int]], labels: tuple[str, ...] | None
) -> np.ndarray:
    panels = [resize_panel(frame, size) for frame, size in zip(frames, panel_sizes, strict=True)]
    if labels is not None:
        panels = [add_label(panel, label) for panel, label in zip(panels, labels, strict=True)]
    return np.hstack(panels)


def create_video(args: argparse.Namespace) -> Path:
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    episode_path = args.episode.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode file not found: {episode_path}")
    if episode_path.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError(f"expected an HDF5 file, got: {episode_path}")

    output_path = args.output.expanduser().resolve() if args.output else default_output_path(episode_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path} (pass --overwrite to replace it)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.stem}.part.mp4")
    if temporary_output.exists():
        temporary_output.unlink()

    camera_paths = [f"observations/images/{key}" for key in args.camera_keys]
    labels = None if args.no_labels else DEFAULT_CAMERA_LABELS
    writer: cv2.VideoWriter | None = None

    try:
        with h5py.File(episode_path, "r") as episode:
            missing = [path for path in camera_paths if path not in episode]
            if missing:
                available = sorted(episode.get("observations/images", {}).keys())
                raise KeyError(
                    f"missing camera datasets: {', '.join(missing)}. "
                    f"Available cameras: {', '.join(available) or '(none)'}"
                )

            datasets = [episode[path] for path in camera_paths]
            frame_counts = [len(dataset) for dataset in datasets]
            if not all(frame_counts):
                raise ValueError("one or more selected camera datasets contain no frames")
            frame_count = min(frame_counts)
            if len(set(frame_counts)) != 1:
                print(
                    f"Warning: camera frame counts differ ({frame_counts}); using the first {frame_count} frames.",
                    file=sys.stderr,
                )

            first_frames = [decode_frame(dataset[0], args.input_color_order) for dataset in datasets]
            target_height = max(frame.shape[0] for frame in first_frames)
            panel_sizes = [panel_dimensions(frame, target_height) for frame in first_frames]
            first_composite = compose_frame(first_frames, panel_sizes, labels)
            height, width = first_composite.shape[:2]
            writer = cv2.VideoWriter(
                str(temporary_output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("could not open the MP4 writer (OpenCV/FFmpeg may lack mp4v support)")

            for index in range(frame_count):
                frames = first_frames if index == 0 else [
                    decode_frame(dataset[index], args.input_color_order) for dataset in datasets
                ]
                writer.write(compose_frame(frames, panel_sizes, labels))
                if (index + 1) % 100 == 0 or index + 1 == frame_count:
                    print(f"Wrote {index + 1}/{frame_count} frames")
    except Exception:
        if writer is not None:
            writer.release()
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    else:
        assert writer is not None
        writer.release()

    temporary_output.replace(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    try:
        output_path = create_video(args)
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
