"""
SurgIQ — Configuration
All paths, model parameters, API keys, and constants live here.
Import this module everywhere else — never hardcode paths or values.
"""

import os
import torch
from pathlib import Path

# ── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Raw data layout
RAW_DIR              = DATA_DIR / "raw"
CHOLECSEG8K_DIR      = RAW_DIR / "cholecseg8k"   # primary dataset

# Legacy Cholec80 paths (kept for reference)
CHOLEC80_VIDEO_DIR   = RAW_DIR / "cholec80" / "videos"
CHOLEC80_TOOL_DIR    = RAW_DIR / "cholec80" / "tool_annotations"
CHOLEC80_PHASE_DIR   = RAW_DIR / "cholec80" / "phase_annotations"
M2CAI_IMG_DIR        = RAW_DIR / "m2cai16" / "images"
M2CAI_ANNOT_DIR      = RAW_DIR / "m2cai16" / "annotations"

# Processed data
FRAMES_DIR              = DATA_DIR / "frames"
YOLO_DATASET_DIR        = DATA_DIR / "processed" / "yolo_dataset"
CLASSIFIER_DATASET_DIR  = DATA_DIR / "processed" / "classifier_dataset"
REPORTS_DIR             = DATA_DIR / "reports"

# Model weights
DETECTOR_WEIGHTS    = MODELS_DIR / "instrument_detector" / "best.pt"
CLASSIFIER_WEIGHTS  = MODELS_DIR / "technique_classifier" / "technique_cnn.pth"
DEEPSORT_CONFIG     = MODELS_DIR / "deepsort" / "config.yaml"

# ── Dataset Splits ────────────────────────────────────────────────────────────
# CRITICAL: TEST_VIDEO_IDS are the held-out test set.
# These 10 videos must NEVER be used during training or validation.
# Only touch them at the very end for final honest evaluation.
TEST_VIDEO_IDS  = list(range(71, 81))   # videos 71-80  (10 videos)
VAL_VIDEO_IDS   = list(range(61, 71))   # videos 61-70  (10 videos)
TRAIN_VIDEO_IDS = list(range(1,  61))   # videos 1-60   (60 videos)

# ── Instrument Classes ────────────────────────────────────────────────────────
INSTRUMENT_CLASSES = [
    "Grasper",      # 0
    "Bipolar",      # 1
    "Hook",         # 2
    "Scissors",     # 3
    "Clipper",      # 4
    "Irrigator",    # 5
    "SpecimenBag",  # 6
]
NUM_INSTRUMENT_CLASSES = len(INSTRUMENT_CLASSES)
INSTRUMENT_TO_IDX = {cls: idx for idx, cls in enumerate(INSTRUMENT_CLASSES)}
IDX_TO_INSTRUMENT = {idx: cls for cls, idx in INSTRUMENT_TO_IDX.items()}

# ── Surgical Phases ───────────────────────────────────────────────────────────
SURGICAL_PHASES = [
    "Preparation",              # 0
    "CalotTriangleDissection",  # 1
    "ClippingCutting",          # 2
    "GallbladderDissection",    # 3
    "GallbladderPackaging",     # 4
    "CleaningCoagulation",      # 5
    "GallbladderRetraction",    # 6
]
NUM_PHASES   = len(SURGICAL_PHASES)
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(SURGICAL_PHASES)}
IDX_TO_PHASE = {idx: phase for phase, idx in PHASE_TO_IDX.items()}

# ── Video / Frame Extraction ──────────────────────────────────────────────────
VIDEO_NATIVE_FPS     = 25   # Cholec80 videos are 25 FPS
FRAME_EXTRACTION_FPS = 1    # Extract 1 frame per second
FRAME_INTERVAL       = VIDEO_NATIVE_FPS // FRAME_EXTRACTION_FPS  # = 25

# ── YOLOv8 Detector ───────────────────────────────────────────────────────────
YOLO_BASE_MODEL     = "yolov8n.pt"  # Nano — fastest; runs well on MPS/CPU
YOLO_IMG_SIZE       = 640
YOLO_EPOCHS         = 50
YOLO_BATCH_SIZE     = 16
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD  = 0.45
YOLO_PATIENCE       = 10            # Early stopping patience

# ── EfficientNet Technique Classifier ─────────────────────────────────────────
CLASSIFIER_IMG_SIZE   = 224
CLASSIFIER_EPOCHS     = 30
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR         = 1e-4
CLASSIFIER_DROPOUT    = 0.3
CLASSIFIER_PATIENCE   = 7           # Early stopping patience

# ── DeepSORT Tracker ──────────────────────────────────────────────────────────
DEEPSORT_MAX_AGE       = 30     # Frames to keep a lost track alive
DEEPSORT_MIN_HITS      = 3      # Hits needed to confirm a track
DEEPSORT_IOU_THRESHOLD = 0.3

# ── Real-Time Pipeline ────────────────────────────────────────────────────────
PIPELINE_FPS              = 5   # Process frames at 5 FPS for the demo
FEEDBACK_INTERVAL_SECONDS = 5   # Call LLM at most every 5 seconds

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST              = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT              = int(os.getenv("REDIS_PORT", 6379))
REDIS_URL               = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}")
REDIS_STREAM_KEY_PREFIX = "surgiq:session"
REDIS_STREAM_MAXLEN     = 1000  # Trim stream to last 1000 messages
DASHBOARD_REFRESH_S     = 0.5   # Streamlit polls Redis every 0.5 s

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama3-70b-8192"
GROQ_MAX_TOKENS = 150

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "surgiq-sessions"

# ── AWS ───────────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET  = os.getenv("S3_BUCKET", "surgiq-models")


# ── Device Selection ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Returns the best available compute device.
    Priority: Apple MPS (M-series Mac) > CUDA > CPU.
    Set PYTORCH_ENABLE_MPS_FALLBACK=1 in your .env to handle
    any ops not yet supported on MPS.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
