"""
SurgIQ — Technique Classifier Training
=======================================
Fine-tunes EfficientNet-B0 (pretrained on ImageNet) on Cholec80 surgical phase
labels.  Runs on Apple MPS backend (M4 Pro) by default; falls back to CUDA or CPU.

Expected dataset layout (built by prepare_dataset.py):
    data/processed/classifier_dataset/
        train/<PhaseA>/*.jpg
        train/<PhaseB>/*.jpg
        val/<PhaseA>/*.jpg
        val/<PhaseB>/*.jpg
        test/...              ← never loaded here

Outputs:
    models/technique_classifier/technique_cnn.pth   — best checkpoint
    models/technique_classifier/training_history.csv

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --epochs 20 --batch-size 16
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ── Data Transforms ───────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Returns image transforms for a given split.
    Training uses augmentation; val/test use only resize + normalise.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats — EfficientNet expects these
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.CLASSIFIER_IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(cfg.CLASSIFIER_IMG_SIZE * 1.15)),
            transforms.CenterCrop(cfg.CLASSIFIER_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, dropout: float = cfg.CLASSIFIER_DROPOUT) -> nn.Module:
    """
    Load pretrained EfficientNet-B0 and replace the classifier head
    for `num_classes` surgical phases.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all layers first, then unfreeze the last two feature blocks
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[6:].parameters():
        param.requires_grad = True

    # Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on val or test set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs    = model(images)
        loss       = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet technique classifier.")
    parser.add_argument("--epochs",     type=int,   default=cfg.CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=cfg.CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.CLASSIFIER_LR)
    parser.add_argument("--dropout",    type=float, default=cfg.CLASSIFIER_DROPOUT)
    parser.add_argument("--patience",   type=int,   default=cfg.CLASSIFIER_PATIENCE,
                        help="Early stopping patience (epochs without val improvement).")
    parser.add_argument("--workers",    type=int,   default=4,
                        help="DataLoader worker count. Use 0 on MPS to avoid issues.")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = cfg.get_device()

    # MPS can have issues with multiprocessing — force workers=0 on MPS
    num_workers = 0 if device.type == "mps" else args.workers

    print("=" * 60)
    print("  SurgIQ — Technique Classifier Training")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Patience   : {args.patience}")
    print("=" * 60)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dir = cfg.CLASSIFIER_DATASET_DIR / "train"
    val_dir   = cfg.CLASSIFIER_DATASET_DIR / "val"

    if not train_dir.exists():
        print(f"ERROR: Training data not found at {train_dir}")
        print("Run prepare_dataset.py --mode classifier first.")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=get_transforms("train"))

    # Guard: check for empty val class dirs before calling ImageFolder
    # (can happen if prepare_dataset.py used a non-stratified split)
    empty_val_classes = []
    for cls in train_dataset.classes:
        cls_val_dir = val_dir / cls
        if not cls_val_dir.exists() or not any(cls_val_dir.glob("*.png")):
            empty_val_classes.append(cls)

    if empty_val_classes:
        print(f"\n  ERROR: Val split is missing images for: {empty_val_classes}")
        print("  Fix: re-run  python training/prepare_dataset.py --skip-yolo")
        print("  (The updated prepare_dataset.py uses a stratified split.)")
        sys.exit(1)

    val_dataset = datasets.ImageFolder(str(val_dir), transform=get_transforms("val"))

    print(f"\n  Classes  : {train_dataset.classes}")
    print(f"  Train    : {len(train_dataset):>6} samples")
    print(f"  Val      : {len(val_dataset):>6} samples\n")

    # Validate class alignment — dataset may use instrument activity labels not surgical phases
    expected = sorted(["no_instrument", "grasper_only", "hook_only", "both_instruments"])
    if sorted(train_dataset.classes) != expected:
        print(f"WARNING: Unexpected dataset classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model, Loss, Optimiser ────────────────────────────────────────────────
    model     = build_model(num_classes=len(train_dataset.classes), dropout=args.dropout)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Output Paths ──────────────────────────────────────────────────────────
    out_dir = cfg.MODELS_DIR / "technique_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path  = out_dir / "technique_cnn.pth"
    history_path       = out_dir / "training_history.csv"

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_acc":      val_acc,
                "classes":      train_dataset.classes,
            }, best_weights_path)

        else:
            patience_count += 1

        flag = " ← best" if is_best else ""
        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"{elapsed:.0f}s{flag}"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if patience_count >= args.patience:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # ── Save Training History ─────────────────────────────────────────────────
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Weights saved     : {best_weights_path}")
    print(f"  History saved     : {history_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
