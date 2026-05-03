"""
SurgIQ — Cholec80 Classifier Dataset Preparation
==================================================
Builds the EfficientNet classifier dataset from Cholec80 (80 videos).
Uses tool presence annotations (Grasper + Hook) to derive 4-class labels.

Unlike CholecSeg8k, Cholec80 has NO segmentation masks — only binary
tool presence labels at 1 fps (every 25 frames for 25fps video).

Labels derived:
    no_instrument    : Grasper=0, Hook=0
    grasper_only     : Grasper=1, Hook=0
    hook_only        : Grasper=0, Hook=1
    both_instruments : Grasper=1, Hook=1

Video-level split (feasible now with 80 videos):
    Test : video48, video52, video55  ← matches CholecSeg8k test set
    Val  : video71 – video80          ← 10 held-out videos
    Train: remaining 67 videos

Frames saved as 256×256 JPEG (classifier resizes to 224 anyway).

Usage:
    python training/prepare_cholec80.py
    python training/prepare_cholec80.py --cholec80-dir ~/Downloads/cholec80
    python training/prepare_cholec80.py --max-videos 20   # quick test run
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ── Constants ─────────────────────────────────────────────────────────────────

CLASSIFIER_LABELS = [
    "no_instrument",
    "grasper_only",
    "hook_only",
    "both_instruments",
]

# These match our CholecSeg8k test split — never touch during training
TEST_VIDEOS = {"video48", "video52", "video55"}

# 10 diverse held-out val videos — true video-level split
VAL_VIDEOS = {
    "video71", "video72", "video73", "video74", "video75",
    "video76", "video77", "video78", "video79", "video80",
}

# Save frames at this size — EfficientNet needs 224, so 256 gives crop room
FRAME_SIZE = 256


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_split(video_name: str) -> str:
    if video_name in TEST_VIDEOS:
        return "test"
    if video_name in VAL_VIDEOS:
        return "val"
    return "train"


def get_label(grasper: int, hook: int) -> str:
    if grasper and hook:
        return "both_instruments"
    elif grasper:
        return "grasper_only"
    elif hook:
        return "hook_only"
    else:
        return "no_instrument"


def read_tool_annotations(ann_path: Path) -> dict[int, str]:
    """
    Parse a Cholec80 tool annotation file.
    Returns dict mapping frame_number → classifier_label.
    """
    labels = {}
    with open(ann_path) as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        frame   = int(parts[0])
        grasper = int(parts[1])
        hook    = int(parts[3])   # Bipolar=2, Hook=3 in column order
        labels[frame] = get_label(grasper, hook)

    return labels


def extract_and_save_frame(
    cap: cv2.VideoCapture,
    frame_idx: int,
    dest_path: Path,
) -> bool:
    """Seek to frame_idx, read it, resize, save as JPEG."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return False
    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE),
                               interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest_path), frame_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_dataset(cholec80_dir: Path, max_videos: int = 0) -> None:
    videos_dir = cholec80_dir / "videos"
    ann_dir    = cholec80_dir / "tool_annotations"

    if not videos_dir.exists():
        print(f"ERROR: videos/ not found at {videos_dir}")
        sys.exit(1)

    # Create output dirs
    out_dir = cfg.CLASSIFIER_DATASET_DIR
    for split in ("train", "val", "test"):
        for label in CLASSIFIER_LABELS:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Find all videos
    video_files = sorted(videos_dir.glob("video*.mp4"))
    if max_videos:
        video_files = video_files[:max_videos]

    print(f"\n  Found {len(video_files)} videos")
    print(f"  Test  : {sorted(TEST_VIDEOS)}")
    print(f"  Val   : video71–video80")
    print(f"  Train : remaining {len(video_files) - len(TEST_VIDEOS) - len(VAL_VIDEOS)} videos\n")

    stats        = defaultdict(lambda: defaultdict(int))
    total_saved  = 0
    total_failed = 0

    for video_path in tqdm(video_files, desc="  Videos"):
        video_name = video_path.stem                    # e.g. "video01"
        ann_path   = ann_dir / f"{video_name}-tool.txt"
        split      = get_split(video_name)

        if not ann_path.exists():
            tqdm.write(f"  WARNING: No annotation for {video_name}, skipping.")
            continue

        # Read annotations
        frame_labels = read_tool_annotations(ann_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"  WARNING: Cannot open {video_path.name}, skipping.")
            continue

        for frame_idx, label in tqdm(
            frame_labels.items(),
            desc=f"    {video_name} ({split})",
            leave=False,
        ):
            fname    = f"{video_name}_frame{frame_idx:06d}.jpg"
            dest     = out_dir / split / label / fname

            if dest.exists():
                continue  # skip if already extracted (resume support)

            ok = extract_and_save_frame(cap, frame_idx, dest)
            if ok:
                stats[split][label] += 1
                total_saved += 1
            else:
                total_failed += 1

        cap.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Cholec80 Classifier Dataset — Summary")
    print(f"{'=' * 60}")
    for split in ("train", "val", "test"):
        print(f"\n  {split.upper()}:")
        split_total = 0
        for label in CLASSIFIER_LABELS:
            count = stats[split][label]
            print(f"    {label:<22}: {count:>6} frames")
            split_total += count
        print(f"    {'TOTAL':<22}: {split_total:>6} frames")

    print(f"\n  Total saved : {total_saved}")
    print(f"  Failed reads: {total_failed}")
    print(f"\n  Output dir  : {out_dir}")
    print(f"\nNext: python training/train_classifier.py --epochs 30")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EfficientNet classifier dataset from Cholec80."
    )
    parser.add_argument(
        "--cholec80-dir",
        type=Path,
        default=Path("~/Downloads/cholec80").expanduser(),
        help="Path to extracted Cholec80 folder (default: ~/Downloads/cholec80)",
    )
    parser.add_argument(
        "--max-videos", type=int, default=0,
        help="Process only first N videos (0 = all). Useful for quick testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  SurgIQ — Cholec80 Dataset Preparation")
    print(f"  Source : {args.cholec80_dir}")
    print(f"  Output : {cfg.CLASSIFIER_DATASET_DIR}")
    print(f"  Videos : {'all 80' if not args.max_videos else args.max_videos}")
    print("=" * 60)

    if not args.cholec80_dir.exists():
        print(f"\nERROR: Cholec80 not found at {args.cholec80_dir}")
        print("Pass the correct path with --cholec80-dir /path/to/cholec80")
        sys.exit(1)

    # Warn if classifier dataset already exists
    if cfg.CLASSIFIER_DATASET_DIR.exists():
        existing = sum(1 for _ in cfg.CLASSIFIER_DATASET_DIR.rglob("*.jpg"))
        if existing > 0:
            print(f"\n  WARNING: {existing} frames already exist in output dir.")
            print("  Script will skip existing files (resume-safe).")
            print("  To start fresh: rm -rf data/processed/classifier_dataset\n")

    build_dataset(args.cholec80_dir, max_videos=args.max_videos)


if __name__ == "__main__":
    main()
