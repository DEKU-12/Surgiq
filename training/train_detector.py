"""
SurgIQ — YOLOv8 Instrument Detector Training
=============================================
Fine-tunes YOLOv8n on the surgical instrument detection dataset
built by prepare_dataset.py.

Intended to run on Google Colab (free GPU) for the full 50-epoch run.
Can also run locally on Mac MPS for debugging / short runs.

See training/colab_training.ipynb for the Colab workflow.

Usage:
    python training/train_detector.py
    python training/train_detector.py --epochs 10 --batch-size 8  # quick debug run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on surgical instruments.")
    parser.add_argument("--epochs",     type=int, default=cfg.YOLO_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.YOLO_BATCH_SIZE)
    parser.add_argument("--img-size",   type=int, default=cfg.YOLO_IMG_SIZE)
    parser.add_argument("--patience",   type=int, default=cfg.YOLO_PATIENCE,
                        help="Early stopping patience.")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from last checkpoint.")
    parser.add_argument("--device",     type=str, default=None,
                        help="Device override: 'cpu', 'mps', '0' (CUDA). "
                             "Auto-detected if omitted.")
    return parser.parse_args()


def detect_device(override: str | None) -> str:
    """
    Return device string for Ultralytics YOLO.
    Ultralytics accepts: 'cpu', 'mps', '0' (first CUDA GPU), etc.
    """
    if override:
        return override
    torch_device = cfg.get_device()
    return str(torch_device)


def main() -> None:
    args   = parse_args()
    device = detect_device(args.device)

    dataset_yaml = cfg.YOLO_DATASET_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"ERROR: YOLO dataset config not found at {dataset_yaml}")
        print("Run prepare_dataset.py --mode yolo first.")
        sys.exit(1)

    out_dir = cfg.MODELS_DIR / "instrument_detector"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SurgIQ — YOLOv8 Detector Training")
    print(f"  Base model  : {cfg.YOLO_BASE_MODEL}")
    print(f"  Dataset     : {dataset_yaml}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Image size  : {args.img_size}")
    print(f"  Device      : {device}")
    print(f"  Output      : {out_dir}")
    print("=" * 60 + "\n")

    # Load base model
    if args.resume and (out_dir / "last.pt").exists():
        print("Resuming from last checkpoint...")
        model = YOLO(str(out_dir / "last.pt"))
    else:
        model = YOLO(cfg.YOLO_BASE_MODEL)

    # Fine-tune
    results = model.train(
        data        = str(dataset_yaml),
        epochs      = args.epochs,
        batch       = args.batch_size,
        imgsz       = args.img_size,
        device      = device,
        patience    = args.patience,
        save        = True,
        save_period = 5,        # Save checkpoint every 5 epochs
        project     = str(out_dir),
        name        = "train",
        exist_ok    = True,
        verbose     = True,
        # Augmentation — helps with surgical lighting variability
        hsv_h       = 0.015,    # Hue jitter
        hsv_s       = 0.7,      # Saturation jitter
        hsv_v       = 0.4,      # Value (brightness) jitter
        flipud      = 0.0,      # No vertical flip (surgery is oriented)
        fliplr      = 0.5,      # Horizontal flip OK
        mosaic      = 1.0,      # Mosaic augmentation
        mixup       = 0.1,      # MixUp augmentation
    )

    # Copy best weights to the expected path
    best_pt = out_dir / "train" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, cfg.DETECTOR_WEIGHTS)
        print(f"\nBest weights copied to: {cfg.DETECTOR_WEIGHTS}")

    print("\nTraining complete.")
    print(f"Results saved to: {out_dir / 'train'}")
    print(f"Upload {cfg.DETECTOR_WEIGHTS} to GitHub Releases as v0.1-models")


if __name__ == "__main__":
    main()
