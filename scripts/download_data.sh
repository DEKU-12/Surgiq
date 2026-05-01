#!/usr/bin/env bash
# SurgIQ — Data Download Helper
# ==============================
# This script does NOT auto-download Cholec80 — it requires email registration.
# It guides you through the manual steps and verifies the expected structure.
#
# Usage:
#   chmod +x scripts/download_data.sh
#   ./scripts/download_data.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/raw"

echo "=================================================="
echo "  SurgIQ — Dataset Setup Guide"
echo "=================================================="

# ── Cholec80 ──────────────────────────────────────────
echo ""
echo "── STEP 1: Download Cholec80 ────────────────────"
echo ""
echo "  1. Visit: http://camma.u-strasbg.fr/datasets"
echo "  2. Fill in the registration form (takes ~24h for approval)"
echo "  3. You'll receive a download link via email"
echo "  4. Download and extract the dataset"
echo ""
echo "  Expected structure after extraction:"
echo "    data/raw/cholec80/"
echo "      videos/          — video01.mp4 ... video80.mp4"
echo "      tool_annotations/ — video01-tool.txt ... video80-tool.txt"
echo "      phase_annotations/ — video01-phase.txt ... video80-phase.txt"
echo ""

CHOLEC80_DIR="$DATA_DIR/cholec80/videos"
if [ -d "$CHOLEC80_DIR" ] && [ "$(ls -A "$CHOLEC80_DIR" 2>/dev/null)" ]; then
    VIDEO_COUNT=$(ls "$CHOLEC80_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Cholec80 videos found: $VIDEO_COUNT videos"
else
    echo "  ✗ Cholec80 videos NOT found at $CHOLEC80_DIR"
    echo "    Place your downloaded videos there and re-run this script."
fi

# ── M2CAI16 ───────────────────────────────────────────
echo ""
echo "── STEP 2: Download M2CAI16 Tool Dataset ────────"
echo ""
echo "  1. Visit: http://camma.u-strasbg.fr/m2cai16"
echo "  2. Download the Tool Localization dataset"
echo "  3. Extract and place under data/raw/m2cai16/"
echo ""
echo "  Expected structure:"
echo "    data/raw/m2cai16/"
echo "      images/      — .jpg frames from M2CAI16 sequences"
echo "      annotations/ — matching .xml files (Pascal VOC bounding boxes)"
echo ""

M2CAI_DIR="$DATA_DIR/m2cai16/annotations"
if [ -d "$M2CAI_DIR" ] && [ "$(ls -A "$M2CAI_DIR" 2>/dev/null)" ]; then
    XML_COUNT=$(ls "$M2CAI_DIR"/*.xml 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ M2CAI16 annotations found: $XML_COUNT XML files"
else
    echo "  ✗ M2CAI16 annotations NOT found at $M2CAI_DIR"
    echo "    NOTE: M2CAI16 is optional but recommended for better YOLO training."
    echo "    Without it, prepare_dataset.py will use Cholec80 pseudo-boxes."
fi

# ── Summary ───────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Once data is in place, run:"
echo ""
echo "  # Build both datasets"
echo "  python training/prepare_dataset.py --mode all --yolo-source m2cai16"
echo ""
echo "  # Or if you only have Cholec80:"
echo "  python training/prepare_dataset.py --mode all --yolo-source cholec80_pseudo"
echo "=================================================="
