# ── Arabic NLP MTL System — Dockerfile ────────────────────────
# Multi-stage build: keeps the final image lean

FROM python:3.10-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p data/processed data/raw checkpoints logs

# Expose API port
EXPOSE 8000

# Default: run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
