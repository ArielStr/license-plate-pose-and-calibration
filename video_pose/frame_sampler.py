from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    """
    Represents one frame sampled from a video.
    """

    frame_index: int
    timestamp_sec: float
    image: np.ndarray


@dataclass(frozen=True)
class VideoMetadata:
    """
    Basic metadata extracted from the video file.
    """

    path: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """
    Read basic metadata without processing the video frames.

    Args:
        video_path:
            Path to the input video.

    Returns:
        VideoMetadata containing FPS, resolution and duration.

    Raises:
        FileNotFoundError:
            If the video file does not exist.

        RuntimeError:
            If OpenCV cannot open the video.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise RuntimeError(
                f"Invalid video FPS returned by OpenCV: {fps}"
            )

        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        return VideoMetadata(
            path=path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_sec=duration_sec,
        )

    finally:
        capture.release()


def sample_video_frames(
    video_path: str | Path,
    every_n_frames: int = 5,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_sampled_frames: int | None = None,
) -> Iterator[SampledFrame]:
    """
    Sample frames from a video at a fixed frame interval.

    For example, with every_n_frames=5, the function returns frames:

        0, 5, 10, 15, ...

    Args:
        video_path:
            Path to the input video.

        every_n_frames:
            Process one frame every N frames.

        start_frame:
            First frame index that may be sampled.

        end_frame:
            Exclusive upper frame boundary.
            None means continue until the end of the video.

        max_sampled_frames:
            Optional maximum number of sampled frames to return.

    Yields:
        SampledFrame objects.

    Raises:
        ValueError:
            If one of the sampling parameters is invalid.

        FileNotFoundError:
            If the video does not exist.

        RuntimeError:
            If OpenCV cannot open or read the video.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be greater than zero")

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")

    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")

    if max_sampled_frames is not None and max_sampled_frames <= 0:
        raise ValueError("max_sampled_frames must be greater than zero")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video FPS returned by OpenCV: {fps}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame_index = start_frame
    sampled_count = 0

    try:
        while True:
            if end_frame is not None and current_frame_index >= end_frame:
                break

            success, frame = capture.read()

            if not success:
                break

            should_sample = (
                (current_frame_index - start_frame) % every_n_frames == 0
            )

            if should_sample:
                timestamp_sec = current_frame_index / fps

                yield SampledFrame(
                    frame_index=current_frame_index,
                    timestamp_sec=timestamp_sec,
                    image=frame,
                )

                sampled_count += 1

                if (
                    max_sampled_frames is not None
                    and sampled_count >= max_sampled_frames
                ):
                    break

            current_frame_index += 1

    finally:
        capture.release()