"""Timestamp-based wrist-camera history sampling helpers."""

import math


def sample_timestamped_history(
    history,
    anchor,
    frame_count,
    temporal_stride,
    camera_fps,
    timeline_anchor_time=None,
):
    """Sample a camera history on a fixed-FPS timeline ending at ``anchor``.

    ``history`` and ``anchor`` contain ``(timestamp_seconds, frame)`` pairs.
    The returned pairs target ``t-(N-1)*stride/fps, ..., t`` and select the
    closest available camera frame for each timestamp. ``timeline_anchor_time``
    may provide a common target shared by multiple cameras, while ``anchor``
    remains the actual current frame that must end this camera's clip. Missing
    history at the beginning of an episode is filled by repeating the earliest
    frame, matching the M2W training loader's boundary behavior.
    """
    frame_count = int(frame_count)
    temporal_stride = int(temporal_stride)
    camera_fps = float(camera_fps)
    if frame_count <= 0:
        return []
    if temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive.")
    if not math.isfinite(camera_fps) or camera_fps <= 0:
        raise ValueError("camera_fps must be a positive finite number.")

    anchor_time, anchor_frame = anchor
    anchor_time = float(anchor_time)
    if timeline_anchor_time is None:
        timeline_anchor_time = anchor_time
    timeline_anchor_time = float(timeline_anchor_time)
    if frame_count == 1:
        return [(anchor_time, anchor_frame)]

    entries = []
    for timestamp, frame in history:
        timestamp = float(timestamp)
        if math.isfinite(timestamp) and timestamp <= anchor_time + 1e-6:
            entries.append((timestamp, frame))

    # The synchronized frame is the authoritative endpoint, even if the raw
    # callback ring was cleared at episode start or overflowed before sampling.
    if not any(frame is anchor_frame for _timestamp, frame in entries):
        entries.append((anchor_time, anchor_frame))
    entries.sort(key=lambda item: item[0])

    timestamps_usable = (
        math.isfinite(anchor_time)
        and math.isfinite(timeline_anchor_time)
        and entries
        and all(math.isfinite(timestamp) for timestamp, _frame in entries)
        and (
            len(entries) == 1
            or entries[-1][0] - entries[0][0] > 1e-9
        )
    )
    if timestamps_usable:
        seconds_per_sample = temporal_stride / camera_fps
        target_times = [
            timeline_anchor_time
            - (frame_count - 1 - index) * seconds_per_sample
            for index in range(frame_count)
        ]
        selected = [
            min(entries, key=lambda item: abs(item[0] - target_time))
            for target_time in target_times
        ]
        selected[-1] = (anchor_time, anchor_frame)
        return selected

    # Fall back to ordered frame indices for cameras that do not publish usable
    # ROS timestamps. This still anchors the clip on the current synchronized
    # frame and retains the configured temporal stride.
    required_span = (frame_count - 1) * temporal_stride + 1
    if len(entries) < required_span:
        entries = [entries[0]] * (required_span - len(entries)) + entries
    selected = entries[-required_span::temporal_stride]
    selected[-1] = (anchor_time, anchor_frame)
    return selected[-frame_count:]
