# SurgIQ — Dockerfile
# ====================
# Multi-stage build:
#   Stage 1 (builder) — install Python deps
#   Stage 2 (runtime) — lean production image

FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# System libs needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code (not data or models — mounted as volumes)
COPY config.py        .
COPY main.py          .
COPY pipeline/        pipeline/
COPY dashboard/       dashboard/
COPY mlops/           mlops/
COPY training/        training/

# Environment defaults (override via docker-compose or --env-file)
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    REDIS_HOST=redis \
    REDIS_PORT=6379

# Expose Streamlit port
EXPOSE 8501

# Default: run the dashboard
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
