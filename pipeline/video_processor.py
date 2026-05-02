"""
SurgIQ — Video Processor
========================
Reads frames from a video file or webcam and yields them at the
configured pipeline FPS.  Handles frame skipping so downstream
components always receive frames at a consistent rate.

Usage:
    from pipeline.video_processor import VideoProcessor

    with VideoProcessor("path/to/video.mp4") as vp:
        for frame_idx, frame in vp:
            # frame is a BGR numpy array (H, W, 3)
            process(frame)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class VideoProcessor:
    """
    Iterates over frames from a video file or webcam.

    Parameters
    ----------
    source : str | int
        Path to a video file, or an integer webcam index (0 = default cam).
    target_fps : int
        How many frames per second to yield. Frames are skipped to match
        this rate relative to the video's native FPS.
    """

    def __init__(self, source: str | int = 0, target_fps: int = cfg.PIPELINE_FPS):
        self.source     = source
        self.target_fps = target_fps
        self.cap        = None
        self._frame_idx = 0

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._skip  = max(1, int(round(native_fps / self.target_fps)))
        total       = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VideoProcessor] source={self.source}")
        print(f"  native_fps={native_fps:.1f}  target_fps={self.target_fps}  skip={self._skip}")
        if total > 0:
            print(f"  total_frames={total}  (~{total / native_fps:.0f}s)")

        return self

    def __exit__(self, *_):
        if self.cap:
            self.cap.release()

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        """
        Returns (frame_idx, frame_bgr).
        Skips frames to match target_fps.
        Raises StopIteration when the source is exhausted.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration

            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Skip frames to match target FPS
            if (current_pos - 1) % self._skip != 0:
                continue

            self._frame_idx += 1
            return self._frame_idx, frame

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height) of the video frames."""
        if self.cap is None:
            raise RuntimeError("VideoProcessor not opened. Use as context manager.")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
