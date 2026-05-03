# SurgIQ — Real-Time AI Surgical Training Coach

> A computer vision pipeline that watches laparoscopic surgery video in real time,
> detects and tracks surgical instruments frame-by-frame, classifies operative
> technique, and delivers LLM-generated coaching feedback to trainee surgeons.

---

## Results

| Model | Metric | Value |
|---|---|---|
| YOLOv8n Instrument Detector | mAP@0.5 | **0.919** |
| YOLOv8n — Grasper | AP@0.5 | **0.986** |
| YOLOv8n — Hook | AP@0.5 | **0.851** |
| EfficientNet-B0 Classifier | Test Accuracy | **85.6%** |
| EfficientNet-B0 Classifier | Val Accuracy | **83.6%** |
| Full Pipeline Latency | Per-frame | ~16ms (63 FPS) |

> **Honest evaluation note:** The classifier was evaluated on 4,837 frames from 3 videos (video48, video52, video55) that were never seen during training — a true video-level held-out test set. Early experiments with only 17 videos yielded 94.7% val accuracy but just 38% test accuracy due to domain shift. Scaling to 80 Cholec80 videos with a proper video-level split resolved this.

---

## Architecture

```
Video / Webcam
      │
      ▼
┌─────────────────┐
│  VideoProcessor  │  ← frame extraction at 5 FPS
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ InstrumentDetect│  ← YOLOv8n  (mAP@0.5 = 0.919)
│      or         │    Grasper · Hook
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ InstrumentTracke│  ← DeepSORT  (stable track IDs)
│      r          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TechniqueClassif│  ← EfficientNet-B0  (94.7% acc)
│     ier         │    no_instrument · grasper_only
└────────┬────────┘    hook_only · both_instruments
         │
         ▼
┌─────────────────┐
│ FeedbackGenerato│  ← Groq / Llama-3.3-70B
│       r         │    rate-limited coaching tips
└────────┬────────┘
         │
         ▼
┌─────────────────┐        ┌──────────────────┐
│  Redis Streams  │───────▶│ Streamlit Dashboa│
│  (pub / sub)    │        │       rd         │
└─────────────────┘        └──────────────────┘
```

---

## Dataset

**Two datasets used:**

**CholecSeg8k** — 17 laparoscopic cholecystectomy video clips with pixel-level segmentation masks. Used for YOLO detector training.

- **Processed (YOLO)**: 5,619 train / 800 val / 1,280 test images
- Key decision: converted pixel-level masks to tight single bounding boxes per instrument class using NumPy argwhere — lifted mAP@0.5 from 0.05 → 0.919

**Cholec80** — 80 laparoscopic cholecystectomy videos with tool presence annotations at 1 FPS. Used for classifier training.

- **Processed (Classifier)**: 158,099 train / 21,562 val / 4,837 test frames
- **Video-level split**: Test={video48, video52, video55}, Val={video71–80}, Train=67 videos
- 4 classes derived from Grasper + Hook presence: `no_instrument`, `grasper_only`, `hook_only`, `both_instruments`

---

## Project Structure

```
SurgIQ/
├── config.py                    # all paths, hyperparams, constants
├── main.py                      # pipeline orchestrator (CLI)
│
├── pipeline/
│   ├── video_processor.py       # frame extraction
│   ├── instrument_detector.py   # YOLOv8n inference
│   ├── instrument_tracker.py    # DeepSORT tracking
│   ├── technique_classifier.py  # EfficientNet-B0 inference
│   ├── feedback_generator.py    # Groq LLM coaching
│   ├── redis_producer.py        # Redis Streams publisher
│   └── redis_consumer.py        # Redis Streams consumer
│
├── training/
│   ├── prepare_dataset.py       # CholecSeg8k → YOLO + classifier datasets
│   ├── train_classifier.py      # EfficientNet-B0 fine-tuning (MPS/CUDA)
│   ├── train_detector.py        # YOLOv8n fine-tuning (Colab/Kaggle)
│   ├── evaluate.py              # mAP + confusion matrix + latency benchmark
│   └── colab_training.ipynb     # Kaggle/Colab YOLO training notebook
│
├── dashboard/
│   └── app.py                   # Streamlit real-time dashboard
│
├── mlops/
│   └── mlflow_logger.py         # experiment tracking
│
├── tests/
│   └── test_pipeline.py         # 25 unit tests (pytest)
│
├── .github/workflows/ci.yml     # GitHub Actions CI
├── Dockerfile                   # production container
└── docker-compose.yml           # full stack (pipeline + dashboard + redis + mlflow)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/DEKU-12/Surgiq.git
cd Surgiq
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Download model weights

Place trained weights in:
```
models/instrument_detector/best.pt      # YOLOv8n (from Kaggle training)
models/technique_classifier/technique_cnn.pth   # EfficientNet-B0
```

### 4. Run the pipeline

```bash
# On a video file
python main.py --source path/to/surgery.mp4

# On a single frame (quick test)
python main.py --source data/raw/cholecseg8k/video01/video01_00080/frame_80_endo.png --no-redis --max-frames 1

# With live display window
python main.py --source video.mp4 --display
```

### 5. Launch the dashboard

```bash
# Terminal 1 — Redis
brew services start redis

# Terminal 2 — Pipeline
python main.py --source video.mp4 --session-id demo

# Terminal 3 — Dashboard
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 6. Docker (full stack)

```bash
docker compose up --build
# Dashboard → http://localhost:8501
# MLflow   → http://localhost:5000
```

---

## Training

### Classifier (runs on Mac M4 / any GPU)

```bash
# 1. Prepare Cholec80 dataset (requires Cholec80 downloaded locally)
python training/prepare_cholec80.py --cholec80-dir ~/Downloads/cholec80

# 2. Train EfficientNet-B0
python training/train_classifier.py --epochs 30

# 3. Evaluate on test set
python training/evaluate.py --split test
```

### Detector (Kaggle / Colab GPU recommended)

Open `training/colab_training.ipynb` on Kaggle with a T4/P100 GPU.
Training takes ~1 hour and saves `best.pt` to the output folder.

---

## Tests

```bash
pytest tests/ -v --cov=pipeline --cov=training
# 25 tests, all passing
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Instrument Detection | YOLOv8n (Ultralytics) |
| Instrument Tracking | DeepSORT |
| Activity Classification | EfficientNet-B0 (PyTorch) |
| LLM Coaching | Groq API / Llama-3.3-70B |
| Message Broker | Redis Streams |
| Dashboard | Streamlit + Plotly |
| Experiment Tracking | MLflow |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Training Hardware | Apple M4 Pro (MPS) + Kaggle T4 GPU + AWS g5.xlarge (A10G) |

---

## Author

**Ayush Sudhir Meshram**
MS Data Science — George Washington University
[GitHub](https://github.com/DEKU-12) · [LinkedIn](https://linkedin.com/in/ayush-meshram)
