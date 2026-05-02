"""
SurgIQ — Instrument Detector
=============================
Wraps YOLOv8n to detect surgical instruments (Grasper, Hook) in a frame.

Returns structured detections that downstream components (tracker,
classifier, feedback generator) can consume directly.

Usage:
    from pipeline.instrument_detector import InstrumentDetector

    detector = InstrumentDetector()
    detections = detector.detect(frame_bgr)
    for det in detections:
        print(det["class_name"], det["confidence"], det["bbox_xyxy"])
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")


class InstrumentDetector:
    """
    YOLOv8n surgical instrument detector.

    Detections are returned as a list of dicts:
        {
            "class_idx"  : int,           # 0=Grasper, 1=Hook
            "class_name" : str,           # "Grasper" or "Hook"
            "confidence" : float,         # 0.0–1.0
            "bbox_xyxy"  : list[float],   # [x1, y1, x2, y2] in pixels
            "bbox_xywhn" : list[float],   # [cx, cy, w, h] normalised
        }
    """

    def __init__(
        self,
        weights_path: Path = cfg.DETECTOR_WEIGHTS,
        conf_threshold: float = cfg.YOLO_CONF_THRESHOLD,
        iou_threshold: float  = cfg.YOLO_IOU_THRESHOLD,
        device: str | None    = None,
    ):
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Detector weights not found: {weights_path}\n"
                "Train the detector first or download best.pt from Kaggle."
            )

        self.model          = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device         = device or str(cfg.get_device())

        print(f"[InstrumentDetector] Loaded weights: {weights_path}")
        print(f"  conf={conf_threshold}  iou={iou_threshold}  device={self.device}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Run YOLOv8n on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR image as returned by cv2.

        Returns
        -------
        list[dict]
            List of detection dicts (see class docstring).
            Empty list if no instruments detected.
        """
        h, w = frame_bgr.shape[:2]

        results = self.model.predict(
            source  = frame_bgr,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            device  = self.device,
            verbose = False,
            imgsz   = cfg.YOLO_IMG_SIZE,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_idx    = int(box.cls[0])
                conf       = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                detections.append({
                    "class_idx"  : cls_idx,
                    "class_name" : cfg.YOLO_CLASS_NAMES[cls_idx] if cls_idx < len(cfg.YOLO_CLASS_NAMES) else f"class_{cls_idx}",
                    "confidence" : conf,
                    "bbox_xyxy"  : [x1, y1, x2, y2],
                    "bbox_xywhn" : [cx, cy, bw, bh],
                })

        return detections

    # ── Annotate ──────────────────────────────────────────────────────────────

    def annotate(self, frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the frame.

        Parameters
        ----------
        frame_bgr   : np.ndarray  Input frame.
        detections  : list[dict]  Output of detect().

        Returns
        -------
        np.ndarray  Annotated frame (copy).
        """
        import cv2
        frame = frame_bgr.copy()

        colors = {
            0: (147, 20,  255),   # Grasper — purple
            1: (0,   200, 255),   # Hook    — yellow
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            color = colors.get(det["class_idx"], (255, 255, 255))
            label = f"{det['class_name']} {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame
