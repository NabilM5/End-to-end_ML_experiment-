# ────────────────────────────────────────────────────────────────────────────
# Dockerfile — project_1: E-Commerce Product Categorisation
# Uses requirements-train.txt (lean) instead of requirements.txt (full).
# Full requirements.txt is for local dev / EDA notebooks only.
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only training dependencies (fast — no Jupyter/matplotlib)
COPY requirements-train.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-train.txt

# Copy source and data
COPY src/   ./src/
COPY data/  ./data/

# Environment defaults
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/output
ENV EXPERIMENT_NAME=project_1_ecom_categorisation

RUN mkdir -p /app/output

# Run full pipeline: preprocess → train
CMD bash -c "\
    echo '=== Step 1: Preprocessing ===' && \
    python src/preprocess.py \
        --train-path ${DATA_DIR}/train.parquet.snappy \
        --test-path  ${DATA_DIR}/test.parquet.snappy  \
        --output-dir ${OUTPUT_DIR} && \
    echo '=== Step 2: Training with MLflow ===' && \
    python src/train.py \
        --data-dir   ${OUTPUT_DIR} \
        --mlflow-uri ${MLFLOW_TRACKING_URI} \
        --experiment ${EXPERIMENT_NAME}"
