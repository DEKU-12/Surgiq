"""
SurgIQ — Instrument Tracker
============================
Wraps deep-sort-realtime to assign consistent track IDs to detected
instruments across frames.  Each instrument keeps the same ID even
when briefly occluded.

Usage:
    from pipeline.instrument_tracker import InstrumentTracker

    tracker = InstrumentTracker()
    tracks  = tracker.update(detections, frame_bgr)
    for track in tracks:
        print(track["track_id"], track["class_name"], track["bbox_xyxy"])
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    raise ImportError(
        "deep-sort-realtime not installed. Run: pip install deep-sort-realtime"
    )


class InstrumentTracker:
    """
    DeepSORT-based multi-object tracker for surgical instruments.

    Tracks are returned as a list of dicts:
        {
            "track_id"   : int,           # stable ID across frames
            "class_idx"  : int,
            "class_name" : str,
            "confidence" : float,
            "bbox_xyxy"  : list[float],   # [x1, y1, x2, y2]
        }
    """

    def __init__(self):
        self.tracker = DeepSort(
            max_age             = cfg.DEEPSORT_MAX_AGE,
            n_init              = cfg.DEEPSORT_MIN_HITS,
            nms_max_overlap     = 1.0,
            max_cosine_distance = 0.4,
            nn_budget           = 100,
            embedder            = "mobilenet",
            half                = False,
            bgr                 = True,
        )
        print(f"[InstrumentTracker] DeepSORT ready  "
              f"max_age={cfg.DEEPSORT_MAX_AGE}  min_hits={cfg.DEEPSORT_MIN_HITS}")

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[dict],
        frame_bgr: np.ndarray,
    ) -> list[dict]:
        """
        Feed new detections into DeepSORT and return confirmed tracks.

        Parameters
        ----------
        detections : list[dict]
            Output of InstrumentDetector.detect().
        frame_bgr  : np.ndarray
            Current frame (needed for appearance embedding).

        Returns
        -------
        list[dict]  Confirmed tracks with stable IDs.
        """
        # Convert to DeepSORT format: ([x1,y1,w,h], conf, class_name)
        raw = []
        class_map = {}   # class_name → class_idx (for reconstruction)

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            w  = x2 - x1
            h  = y2 - y1
            raw.append(([x1, y1, w, h], det["confidence"], det["class_name"]))
            class_map[det["class_name"]] = det["class_idx"]

        ds_tracks = self.tracker.update_tracks(raw, frame=frame_bgr)

        tracks = []
        for t in ds_tracks:
            if not t.is_confirmed():
                continue

            ltrb       = t.to_ltrb()   # [x1, y1, x2, y2]
            class_name = t.det_class or "Unknown"
            class_idx  = class_map.get(class_name, -1)

            tracks.append({
                "track_id"   : t.track_id,
                "class_idx"  : class_idx,
                "class_name" : class_name,
                "confidence" : t.det_conf or 0.0,
                "bbox_xyxy"  : [float(v) for v in ltrb],
            })

        return tracks

    def reset(self) -> None:
        """Reset tracker state (call between unrelated video segments)."""
        self.tracker.delete_all_tracks()
