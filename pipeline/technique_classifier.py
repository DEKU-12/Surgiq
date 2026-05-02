"""
SurgIQ — Technique Classifier
==============================
Runs the trained EfficientNet-B0 model on a frame to classify which
instruments are currently active.

4 output classes:
    0 → no_instrument
    1 → grasper_only
    2 → hook_only
    3 → both_instruments

Usage:
    from pipeline.technique_classifier import TechniqueClassifier

    clf = TechniqueClassifier()
    result = clf.classify(frame_bgr)
    print(result["label"], result["confidence"])
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class TechniqueClassifier:
    """
    EfficientNet-B0 instrument activity classifier.

    classify() returns a dict:
        {
            "label"       : str,    # e.g. "hook_only"
            "class_idx"   : int,    # 0–3
            "confidence"  : float,  # 0.0–1.0
            "probabilities": dict,  # {label: prob} for all classes
        }
    """

    # Class labels in training order (ImageFolder sorts alphabetically)
    CLASSES = ["both_instruments", "grasper_only", "hook_only", "no_instrument"]

    def __init__(
        self,
        weights_path: Path = cfg.CLASSIFIER_WEIGHTS,
        device: torch.device | None = None,
    ):
        self.device = device or cfg.get_device()

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Classifier weights not found: {weights_path}\n"
                "Train the classifier first: python training/train_classifier.py"
            )

        # Build model and load weights
        self.model = self._load_model(weights_path)
        self.model.eval()

        # Inference transform (same as val transform in training)
        self.transform = transforms.Compose([
            transforms.Resize(int(cfg.CLASSIFIER_IMG_SIZE * 1.15)),
            transforms.CenterCrop(cfg.CLASSIFIER_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        print(f"[TechniqueClassifier] Loaded weights: {weights_path}")
        print(f"  device={self.device}  classes={self.CLASSES}")

    def _load_model(self, weights_path: Path) -> nn.Module:
        checkpoint   = torch.load(weights_path, map_location=self.device)
        saved_classes = checkpoint.get("classes", self.CLASSES)

        # Use classes from checkpoint if available
        if saved_classes:
            self.CLASSES = saved_classes

        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=cfg.CLASSIFIER_DROPOUT, inplace=True),
            nn.Linear(in_features, len(self.CLASSES)),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        return model

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def classify(self, frame_bgr: np.ndarray) -> dict:
        """
        Classify instrument activity in a single frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR frame from cv2.

        Returns
        -------
        dict  with keys: label, class_idx, confidence, probabilities
        """
        import cv2
        # BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)
        tensor    = self.transform(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        class_idx  = int(probs.argmax())
        confidence = float(probs[class_idx])
        label      = self.CLASSES[class_idx]

        return {
            "label"        : label,
            "class_idx"    : class_idx,
            "confidence"   : confidence,
            "probabilities": {cls: float(p) for cls, p in zip(self.CLASSES, probs)},
        }
