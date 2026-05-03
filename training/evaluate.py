"""
SurgIQ — Model Evaluation
==========================
Evaluates both trained models on the VALIDATION set during development,
and on the TEST set for final honest evaluation.

IMPORTANT: Only run --split test after ALL training is complete.
           The test split (videos 71-80) must remain untouched until then.

Metrics produced:
  Detector   : mAP@0.5, mAP@0.5:0.95, per-class AP
  Classifier : accuracy, per-class precision/recall/F1, confusion matrix

Outputs:
  data/reports/detector_metrics.json
  data/reports/classifier_metrics.json
  data/reports/confusion_matrix.png
  data/reports/evaluation_summary.txt

Usage:
    # Validate during development (safe — uses val split)
    python training/evaluate.py --split val

    # Final honest evaluation (only run once, at project end)
    python training/evaluate.py --split test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_classifier(weights_path: Path, num_classes: int, device: torch.device):
    """Load a saved EfficientNet checkpoint."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    import torch.nn as nn
    model.classifier = nn.Sequential(
        nn.Dropout(p=cfg.CLASSIFIER_DROPOUT, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("classes", cfg.SURGICAL_PHASES)


# ── Detector Evaluation ───────────────────────────────────────────────────────

def evaluate_detector(split: str) -> dict:
    """
    Run YOLOv8 validation on the requested split.
    Returns metrics dict with mAP values.
    """
    print(f"\n── Detector Evaluation ({split}) ──────────────────────────────────")

    if not cfg.DETECTOR_WEIGHTS.exists():
        print(f"ERROR: Detector weights not found at {cfg.DETECTOR_WEIGHTS}")
        print("Train the detector first: python training/train_detector.py")
        return {}

    dataset_yaml = cfg.YOLO_DATASET_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"ERROR: dataset.yaml not found at {dataset_yaml}")
        return {}

    model   = YOLO(str(cfg.DETECTOR_WEIGHTS))
    results = model.val(
        data   = str(dataset_yaml),
        split  = split,
        imgsz  = cfg.YOLO_IMG_SIZE,
        conf   = cfg.YOLO_CONF_THRESHOLD,
        iou    = cfg.YOLO_IOU_THRESHOLD,
        device = str(cfg.get_device()),
        verbose= True,
    )

    metrics = {
        "split":           split,
        "mAP50":           float(results.box.map50),
        "mAP50_95":        float(results.box.map),
        "precision":       float(results.box.mp),
        "recall":          float(results.box.mr),
        "per_class_AP50":  {
            cfg.IDX_TO_INSTRUMENT[i]: float(ap)
            for i, ap in enumerate(results.box.ap50)
        },
    }

    print(f"\n  mAP@0.5      : {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics['mAP50_95']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print("\n  Per-class AP@0.5:")
    for cls_name, ap in metrics["per_class_AP50"].items():
        print(f"    {cls_name:<14}: {ap:.4f}")

    return metrics


# ── Classifier Evaluation ─────────────────────────────────────────────────────

def evaluate_classifier(split: str) -> dict:
    """
    Run EfficientNet evaluation on the requested split.
    Returns metrics dict; also saves the confusion matrix PNG.
    """
    print(f"\n── Classifier Evaluation ({split}) ────────────────────────────────")

    if not cfg.CLASSIFIER_WEIGHTS.exists():
        print(f"ERROR: Classifier weights not found at {cfg.CLASSIFIER_WEIGHTS}")
        print("Train the classifier first: python training/train_classifier.py")
        return {}

    split_dir = cfg.CLASSIFIER_DATASET_DIR / split
    if not split_dir.exists():
        print(f"ERROR: Classifier {split} data not found at {split_dir}")
        return {}

    device = cfg.get_device()

    val_transform = transforms.Compose([
        transforms.Resize(int(cfg.CLASSIFIER_IMG_SIZE * 1.15)),
        transforms.CenterCrop(cfg.CLASSIFIER_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Skip empty class directories — some test videos may lack certain classes
    empty_classes = [
        d.name for d in split_dir.iterdir()
        if d.is_dir() and not any(d.iterdir())
    ]
    if empty_classes:
        print(f"\n  WARNING: Skipping empty class dirs in {split}: {empty_classes}")
        print("  This is expected if test videos don't cover all instrument combinations.")
        # Temporarily remove empty dirs from ImageFolder by filtering
        import shutil, tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir() and any(cls_dir.iterdir()):
                (tmp_dir / cls_dir.name).symlink_to(cls_dir.resolve())
        split_dir = tmp_dir

    dataset = datasets.ImageFolder(str(split_dir), transform=val_transform)
    loader  = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=0 if device.type == "mps" else 4,
    )

    # Always load num_classes from checkpoint — dataset may have fewer classes (e.g. empty test dirs)
    ckpt        = torch.load(cfg.CLASSIFIER_WEIGHTS, map_location=device)
    num_classes = len(ckpt.get("classes", ["no_instrument", "grasper_only", "hook_only", "both_instruments"]))
    model, class_names = load_classifier(cfg.CLASSIFIER_WEIGHTS, num_classes, device)
    # Remap dataset class indices to match the full model class list
    dataset_to_model = [class_names.index(c) for c in dataset.classes]

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  evaluating"):
            outputs = model(images.to(device))
            preds   = outputs.argmax(dim=1).cpu().numpy()
            # Remap dataset label indices → model label indices
            remapped_labels = [dataset_to_model[l] for l in labels.numpy()]
            all_preds.extend(preds)
            all_labels.extend(remapped_labels)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Only report on classes actually present in this split
    present_indices = sorted(set(all_labels))
    present_names   = [class_names[i] for i in present_indices]

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=present_indices,
        target_names=present_names,
        output_dict=True,
    )

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"\n  {classification_report(all_labels, all_preds, labels=present_indices, target_names=present_names)}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = cfg.REPORTS_DIR / f"confusion_matrix_{split}.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Phase")
    ax.set_ylabel("True Phase")
    ax.set_title(f"Phase Classifier Confusion Matrix ({split})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

    metrics = {
        "split":           split,
        "accuracy":        float(acc),
        "classification_report": report,
        "confusion_matrix":      cm.tolist(),
        "class_names":           class_names,
    }

    return metrics


# ── Latency Benchmark ─────────────────────────────────────────────────────────

def benchmark_latency(n_frames: int = 100) -> dict:
    """
    Measures per-frame latency for detector + classifier on random input.
    Returns dict with avg_ms, p95_ms latency values.
    """
    print("\n── Latency Benchmark ──────────────────────────────────────────────")

    import time
    device = cfg.get_device()
    times  = {"detector": [], "classifier": []}

    # ── Detector latency ──────────────────────────────────────────────────────
    if cfg.DETECTOR_WEIGHTS.exists():
        yolo_model = YOLO(str(cfg.DETECTOR_WEIGHTS))
        dummy_img  = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        print(f"  Running {n_frames} detector inference passes...")
        for _ in tqdm(range(n_frames), leave=False):
            t0 = time.perf_counter()
            yolo_model.predict(dummy_img, verbose=False, device=str(device))
            times["detector"].append((time.perf_counter() - t0) * 1000)

        det_avg = np.mean(times["detector"])
        det_p95 = np.percentile(times["detector"], 95)
        print(f"  Detector  — avg: {det_avg:.1f}ms  p95: {det_p95:.1f}ms")
    else:
        det_avg = det_p95 = 0.0
        print("  Detector weights not found — skipping.")

    # ── Classifier latency ────────────────────────────────────────────────────
    if cfg.CLASSIFIER_WEIGHTS.exists():
        # Read num_classes from checkpoint to avoid hardcoded mismatch
        checkpoint  = torch.load(cfg.CLASSIFIER_WEIGHTS, map_location=device)
        num_classes = len(checkpoint.get("classes", ["no_instrument", "grasper_only", "hook_only", "both_instruments"]))
        model, _ = load_classifier(cfg.CLASSIFIER_WEIGHTS, num_classes, device)
        dummy_tensor = torch.rand(1, 3, cfg.CLASSIFIER_IMG_SIZE, cfg.CLASSIFIER_IMG_SIZE).to(device)

        print(f"  Running {n_frames} classifier inference passes...")
        with torch.no_grad():
            for _ in tqdm(range(n_frames), leave=False):
                t0 = time.perf_counter()
                model(dummy_tensor)
                times["classifier"].append((time.perf_counter() - t0) * 1000)

        cls_avg = np.mean(times["classifier"])
        cls_p95 = np.percentile(times["classifier"], 95)
        print(f"  Classifier— avg: {cls_avg:.1f}ms  p95: {cls_p95:.1f}ms")
    else:
        cls_avg = cls_p95 = 0.0
        print("  Classifier weights not found — skipping.")

    total_avg = det_avg + cls_avg
    total_p95 = det_p95 + cls_p95
    print(f"\n  Combined  — avg: {total_avg:.1f}ms  p95: {total_p95:.1f}ms")
    print(f"  Effective throughput (avg): {1000 / total_avg:.1f} FPS" if total_avg > 0 else "")

    return {
        "detector_avg_ms":    det_avg,
        "detector_p95_ms":    det_p95,
        "classifier_avg_ms":  cls_avg,
        "classifier_p95_ms":  cls_p95,
        "combined_avg_ms":    total_avg,
        "combined_p95_ms":    total_p95,
    }


# ── Save Results ──────────────────────────────────────────────────────────────

def save_metrics(detector_metrics: dict, classifier_metrics: dict, latency: dict, split: str) -> None:
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON reports
    if detector_metrics:
        path = cfg.REPORTS_DIR / f"detector_metrics_{split}.json"
        with open(path, "w") as f:
            json.dump(detector_metrics, f, indent=2)
        print(f"\n  Detector metrics saved  : {path}")

    if classifier_metrics:
        path = cfg.REPORTS_DIR / f"classifier_metrics_{split}.json"
        with open(path, "w") as f:
            json.dump(classifier_metrics, f, indent=2)
        print(f"  Classifier metrics saved: {path}")

    # Human-readable summary
    summary_path = cfg.REPORTS_DIR / f"evaluation_summary_{split}.txt"
    with open(summary_path, "w") as f:
        f.write(f"SurgIQ — Evaluation Summary ({split.upper()} SET)\n")
        f.write("=" * 60 + "\n\n")

        if detector_metrics:
            f.write("INSTRUMENT DETECTOR (YOLOv8n)\n")
            f.write(f"  mAP@0.5      : {detector_metrics.get('mAP50', 0):.4f}\n")
            f.write(f"  mAP@0.5:0.95 : {detector_metrics.get('mAP50_95', 0):.4f}\n")
            f.write(f"  Precision    : {detector_metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall       : {detector_metrics.get('recall', 0):.4f}\n")
            f.write("\n  Per-class AP@0.5:\n")
            for cls, ap in detector_metrics.get("per_class_AP50", {}).items():
                f.write(f"    {cls:<14}: {ap:.4f}\n")

        if classifier_metrics:
            f.write("\nTECHNIQUE CLASSIFIER (EfficientNet-B0)\n")
            f.write(f"  Accuracy     : {classifier_metrics.get('accuracy', 0):.4f}\n")

        if latency:
            f.write("\nLATENCY\n")
            f.write(f"  Detector avg  : {latency.get('detector_avg_ms', 0):.1f}ms\n")
            f.write(f"  Classifier avg: {latency.get('classifier_avg_ms', 0):.1f}ms\n")
            f.write(f"  Combined avg  : {latency.get('combined_avg_ms', 0):.1f}ms\n")
            f.write(f"  Combined p95  : {latency.get('combined_p95_ms', 0):.1f}ms\n")

    print(f"  Summary saved           : {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SurgIQ models.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate on. Default: val. "
             "Only use 'test' for final honest evaluation at project end.",
    )
    parser.add_argument(
        "--skip-latency", action="store_true",
        help="Skip latency benchmarking."
    )
    parser.add_argument(
        "--latency-frames", type=int, default=100,
        help="Number of frames to use for latency benchmarking. Default: 100"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.split == "test":
        print("\n" + "!" * 68)
        print("  WARNING: You are evaluating on the TEST set.")
        print("  This should only happen ONCE, at the very end of the project.")
        print("  Make sure all training is complete before proceeding.")
        print("!" * 68)
        confirm = input("\n  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    print("\n" + "=" * 68)
    print(f"  SurgIQ — Model Evaluation  ({args.split.upper()} split)")
    print("=" * 68)

    detector_metrics   = evaluate_detector(args.split)
    classifier_metrics = evaluate_classifier(args.split)
    latency            = {} if args.skip_latency else benchmark_latency(args.latency_frames)

    save_metrics(detector_metrics, classifier_metrics, latency, args.split)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
