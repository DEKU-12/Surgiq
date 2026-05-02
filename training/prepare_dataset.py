"""
SurgIQ — Dataset Preparation (CholecSeg8k)
==========================================
Prepares two datasets from CholecSeg8k for training:

  1. YOLO detection dataset  — frames + bounding boxes converted from
                               segmentation masks (Grasper + Hook)

  2. Classifier dataset      — frames organised by instrument activity
                               class (4 classes based on which instruments
                               are visible in each frame)

CholecSeg8k structure:
    data/raw/cholecseg8k/
        video01/
            video01_00080/
                frame_XX_endo.png            ← surgical image
                frame_XX_endo_mask.png       ← grayscale mask (pixel = class id)
                frame_XX_endo_color_mask.png ← colour mask (visualisation only)
                frame_XX_endo_watershed_mask.png

CholecSeg8k instrument class pixel values:
    5  → Grasper
    9  → L-hook Electrocautery (Hook)

Classifier labels (derived from instrument presence):
    0 → no_instrument   (neither visible)
    1 → grasper_only    (only Grasper visible)
    2 → hook_only       (only Hook visible)
    3 → both_instruments (both visible)

Dataset split (17 videos total, hard-coded):
    Test  : video48, video52, video55  (3 videos — NEVER touch during training)
    Val   : video35, video37, video43  (3 videos)
    Train : remaining 11 videos

Usage:
    python training/prepare_dataset.py
    python training/prepare_dataset.py --min-area 200 --skip-yolo
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# ── CholecSeg8k Constants ─────────────────────────────────────────────────────

CHOLECSEG8K_DIR = cfg.RAW_DIR / "cholecseg8k"

# Pixel value in the grayscale mask for each instrument
# Confirmed by matching color masks to known CholecSeg8k palette:
#   val=13 → BGR(160,78,232) → RGB(232,78,160) → Grasper (purple)
#   val=12 → BGR(75,183,186) → RGB(186,183,75) → L-hook  (yellow-green)
MASK_GRASPER = 13
MASK_HOOK    = 12

# Map mask pixel value → YOLO class index
MASK_TO_YOLO_CLASS = {
    MASK_GRASPER: 0,   # Grasper → class 0
    MASK_HOOK:    1,   # Hook    → class 1
}

YOLO_CLASS_NAMES = ["Grasper", "Hook"]

# Video-level held-out test set — NEVER used during training
TEST_VIDEOS  = {"video48", "video52", "video55"}
# All other videos use frame-level 80/10/10 split for train/val
# This ensures instrument frames appear in all splits
FRAME_TRAIN_RATIO = 0.80
FRAME_VAL_RATIO   = 0.10
# Remaining 0.10 goes to val within non-test videos (no val from test videos)

# Classifier label names
CLASSIFIER_LABELS = [
    "no_instrument",
    "grasper_only",
    "hook_only",
    "both_instruments",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def split_for_video(video_name: str) -> str:
    """Returns 'test' for held-out videos, else 'train' (frame-level split handles val)."""
    if video_name in TEST_VIDEOS:
        return "test"
    return "train_val"  # will be split at frame level


def frame_split(frame_idx: int, total_frames: int) -> str:
    """
    Assigns a frame to train or val based on its position within the video.
    Last 10% of frames → val, rest → train.
    NOTE: Used only for YOLO dataset. Classifier uses stratified split.
    """
    val_start = int(total_frames * FRAME_TRAIN_RATIO)
    if frame_idx >= val_start:
        return "val"
    return "train"


def mask_to_bboxes(
    mask_path: Path,
    min_area: int = 150,
) -> list[tuple[int, float, float, float, float]]:
    """
    Read a CholecSeg8k grayscale mask and convert instrument pixels
    to YOLO bounding boxes.

    Returns ONE bounding box per instrument class — the tight rectangle
    that encloses ALL pixels of that class.  This avoids flooding the
    model with dozens of tiny connected-component fragments per frame.

    Returns list of (class_idx, cx, cy, w, h) normalised to [0, 1].
    Classes with fewer than min_area total pixels are skipped (noise).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape
    annotations = []

    for pixel_val, class_idx in MASK_TO_YOLO_CLASS.items():
        # Binary mask for this instrument
        binary = (mask == pixel_val)
        total_pixels = binary.sum()
        if total_pixels < min_area:
            continue

        # Single tight bbox enclosing ALL pixels of this class
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        bw = int(x_max - x_min + 1)
        bh = int(y_max - y_min + 1)

        cx_n = (x_min + bw / 2) / w
        cy_n = (y_min + bh / 2) / h
        bw_n = bw / w
        bh_n = bh / h

        # Clamp to [0, 1]
        cx_n = max(0.0, min(1.0, cx_n))
        cy_n = max(0.0, min(1.0, cy_n))
        bw_n = max(0.001, min(1.0, bw_n))
        bh_n = max(0.001, min(1.0, bh_n))

        annotations.append((class_idx, cx_n, cy_n, bw_n, bh_n))

    return annotations


def get_classifier_label(annotations: list) -> str:
    """
    Derive classifier label from which instruments appear in a frame.
    """
    classes = {ann[0] for ann in annotations}
    has_grasper = 0 in classes
    has_hook    = 1 in classes

    if has_grasper and has_hook:
        return "both_instruments"
    elif has_grasper:
        return "grasper_only"
    elif has_hook:
        return "hook_only"
    else:
        return "no_instrument"


# ── YOLO Dataset Builder ──────────────────────────────────────────────────────

def build_yolo_dataset(min_area: int = 150) -> None:
    """
    Walk CholecSeg8k, convert segmentation masks to YOLO bounding boxes,
    and copy images + labels into the YOLO dataset directory.
    """
    print("\n[1/2] Building YOLO detection dataset from CholecSeg8k masks...")

    if not CHOLECSEG8K_DIR.exists():
        print(f"  ERROR: CholecSeg8k not found at {CHOLECSEG8K_DIR}")
        sys.exit(1)

    for split in ("train", "val", "test"):
        make_dirs(
            cfg.YOLO_DATASET_DIR / "images" / split,
            cfg.YOLO_DATASET_DIR / "labels" / split,
        )

    stats          = {"train": 0, "val": 0, "test": 0}
    skipped_nobox  = 0

    video_dirs = sorted([d for d in CHOLECSEG8K_DIR.iterdir() if d.is_dir()])

    for video_dir in tqdm(video_dirs, desc="  Videos"):
        video_name  = video_dir.name
        video_split = split_for_video(video_name)
        frame_dirs  = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        total_frame_dirs = len(frame_dirs)

        for fd_idx, frame_dir in enumerate(frame_dirs):
            image_files = sorted(frame_dir.glob("*_endo.png"))
            total_imgs  = len(image_files)

            for img_idx, img_path in enumerate(image_files):
                stem      = img_path.stem
                mask_path = frame_dir / (stem + "_mask.png")
                if not mask_path.exists():
                    continue

                annotations = mask_to_bboxes(mask_path, min_area=min_area)

                # Determine split
                if video_split == "test":
                    split = "test"
                else:
                    # Frame-level split within non-test videos
                    global_idx = fd_idx * 1000 + img_idx
                    global_total = total_frame_dirs * 1000
                    split = frame_split(global_idx, global_total)

                # Skip frames with no instruments for train/val
                if split != "test" and not annotations:
                    skipped_nobox += 1
                    continue

                unique_stem = f"{video_name}_{frame_dir.name}_{stem}"
                dest_img    = cfg.YOLO_DATASET_DIR / "images" / split / f"{unique_stem}.png"
                dest_label  = cfg.YOLO_DATASET_DIR / "labels" / split / f"{unique_stem}.txt"

                shutil.copy2(img_path, dest_img)
                with open(dest_label, "w") as f:
                    for cls_idx, cx, cy, bw, bh in annotations:
                        f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                stats[split] += 1

    # Write dataset.yaml
    yaml_path = cfg.YOLO_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""# SurgIQ — YOLOv8 Dataset (CholecSeg8k)
path: {cfg.YOLO_DATASET_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(YOLO_CLASS_NAMES)}
names: {YOLO_CLASS_NAMES}
""")

    print(f"\n  YOLO dataset built:")
    for split, count in stats.items():
        print(f"    {split:>5}: {count:>6} images")
    print(f"    skipped (no instrument): {skipped_nobox}")
    print(f"  dataset.yaml → {yaml_path}")


# ── Classifier Dataset Builder ────────────────────────────────────────────────

def build_classifier_dataset(min_area: int = 150) -> None:
    """
    Build EfficientNet training dataset organised by instrument activity class.

    Uses a STRATIFIED split — every class is guaranteed to appear in train AND val.
    All frames are collected first, grouped by class, then randomly split 80/20
    (of the non-test pool).  Test videos are always held out entirely.

    Structure:
        data/processed/classifier_dataset/
            train/
                no_instrument/
                grasper_only/
                hook_only/
                both_instruments/
            val/
            test/
    """
    import random
    from collections import defaultdict

    print("\n[2/2] Building classifier dataset from CholecSeg8k (stratified split)...")

    if not CHOLECSEG8K_DIR.exists():
        print(f"  ERROR: CholecSeg8k not found at {CHOLECSEG8K_DIR}")
        sys.exit(1)

    for split in ("train", "val", "test"):
        for label in CLASSIFIER_LABELS:
            make_dirs(cfg.CLASSIFIER_DATASET_DIR / split / label)

    stats = {"train": 0, "val": 0, "test": 0}

    # ── Pass 1: collect all frames with their labels ──────────────────────────
    test_frames    = []   # (img_path, label, unique_name) — go straight to test
    trainval_frames = defaultdict(list)  # label → [(img_path, unique_name)]

    video_dirs = sorted([d for d in CHOLECSEG8K_DIR.iterdir() if d.is_dir()])

    for video_dir in tqdm(video_dirs, desc="  Scanning"):
        video_name  = video_dir.name
        is_test     = video_name in TEST_VIDEOS
        frame_dirs  = sorted([d for d in video_dir.iterdir() if d.is_dir()])

        for frame_dir in frame_dirs:
            for img_path in sorted(frame_dir.glob("*_endo.png")):
                stem      = img_path.stem
                mask_path = frame_dir / (stem + "_mask.png")
                if not mask_path.exists():
                    continue

                annotations = mask_to_bboxes(mask_path, min_area=min_area)
                label       = get_classifier_label(annotations)
                unique_name = f"{video_name}_{frame_dir.name}_{stem}.png"

                if is_test:
                    test_frames.append((img_path, label, unique_name))
                else:
                    trainval_frames[label].append((img_path, unique_name))

    # ── Pass 2: stratified 80/20 split on non-test frames ────────────────────
    random.seed(42)
    train_frames = []
    val_frames   = []

    for label, items in trainval_frames.items():
        random.shuffle(items)
        val_size = max(1, int(len(items) * FRAME_VAL_RATIO))  # at least 1 per class
        val_frames.extend([(img_path, label, name) for img_path, name in items[:val_size]])
        train_frames.extend([(img_path, label, name) for img_path, name in items[val_size:]])

    # ── Pass 3: copy files to destinations ───────────────────────────────────
    for frames, split in [(train_frames, "train"), (val_frames, "val"), (test_frames, "test")]:
        for img_path, label, unique_name in tqdm(frames, desc=f"  Copying {split}", leave=False):
            dest = cfg.CLASSIFIER_DATASET_DIR / split / label / unique_name
            shutil.copy2(img_path, dest)
            stats[split] += 1

    print(f"\n  Classifier dataset built (stratified split):")
    for split in ("train", "val", "test"):
        split_dir = cfg.CLASSIFIER_DATASET_DIR / split
        print(f"\n  {split}:")
        for label in CLASSIFIER_LABELS:
            count = len(list((split_dir / label).glob("*.png")))
            print(f"    {label:<20}: {count:>5} frames")
        print(f"    {'TOTAL':<20}: {stats[split]:>5} frames")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    print("\n── Dataset Summary ──────────────────────────────────────────────────")
    print(f"\n  Test videos  (held-out): {sorted(TEST_VIDEOS)}")
    print(f"  Split strategy          : frame-level 80/10 train/val within non-test videos\n")

    if cfg.YOLO_DATASET_DIR.exists():
        print("  YOLO dataset:")
        for split in ("train", "val", "test"):
            img_dir = cfg.YOLO_DATASET_DIR / "images" / split
            count = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            print(f"    {split:>5}: {count:>6} images")

    if cfg.CLASSIFIER_DATASET_DIR.exists():
        print("\n  Classifier dataset:")
        for split in ("train", "val", "test"):
            count = sum(
                1 for _ in (cfg.CLASSIFIER_DATASET_DIR / split).rglob("*.png")
            ) if (cfg.CLASSIFIER_DATASET_DIR / split).exists() else 0
            print(f"    {split:>5}: {count:>6} frames")

    print("\n  Instrument classes for YOLO:")
    for idx, name in enumerate(YOLO_CLASS_NAMES):
        print(f"    {idx}: {name}")

    print("\n  Classifier labels:")
    for idx, label in enumerate(CLASSIFIER_LABELS):
        print(f"    {idx}: {label}")
    print("────────────────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SurgIQ datasets from CholecSeg8k."
    )
    parser.add_argument(
        "--min-area", type=int, default=150,
        help="Minimum pixel area for a mask component to become a bounding box. "
             "Lower = more boxes but more noise. Default: 150"
    )
    parser.add_argument(
        "--skip-yolo", action="store_true",
        help="Skip YOLO dataset building."
    )
    parser.add_argument(
        "--skip-classifier", action="store_true",
        help="Skip classifier dataset building."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 68)
    print("  SurgIQ — Dataset Preparation (CholecSeg8k)")
    print(f"  Source      : {CHOLECSEG8K_DIR}")
    print(f"  Min bbox area: {args.min_area} px")
    print(f"  Test videos : {sorted(TEST_VIDEOS)}  ← HELD OUT")
    print(f"  Split       : frame-level 80/10/10 train/val/test")
    print("=" * 68)

    if not args.skip_yolo:
        build_yolo_dataset(min_area=args.min_area)

    if not args.skip_classifier:
        build_classifier_dataset(min_area=args.min_area)

    print_summary()
    print("Dataset preparation complete.")
    print(f"\nNext step: python training/train_classifier.py")


if __name__ == "__main__":
    main()
