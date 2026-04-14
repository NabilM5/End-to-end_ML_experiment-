#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# run_experiment.sh — End-to-end ML experiment for project_1
#
# Modes:
#   ./run_experiment.sh            # run locally (needs Python + requirements)
#   ./run_experiment.sh --docker   # run via Docker Compose
#
# Local prerequisites:
#   pip install -r requirements.txt
#
# Docker prerequisites:
#   docker, docker compose
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail   # exit on error, undefined var, pipe failure

# ── Default configuration ─────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-project_1_ecom_categorisation}"

USE_DOCKER=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --docker)      USE_DOCKER=true ;;
        --data-dir)    DATA_DIR="$2";    shift ;;
        --output-dir)  OUTPUT_DIR="$2";  shift ;;
        --mlflow-uri)  MLFLOW_URI="$2";  shift ;;
        --experiment)  EXPERIMENT_NAME="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--docker] [--data-dir PATH] [--output-dir PATH]"
            echo "          [--mlflow-uri URI] [--experiment NAME]"
            exit 0 ;;
        *)
            echo "Unknown argument: $1  (use --help for usage)" >&2
            exit 1 ;;
    esac
    shift
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  project_1 — E-Commerce Product Categorisation"
echo "  MLOps Course"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
date
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER MODE — delegates everything to docker compose
# ─────────────────────────────────────────────────────────────────────────────
if [ "$USE_DOCKER" = true ]; then
    echo "▶  Docker mode selected."
    echo ""

    # Check docker is available
    if ! command -v docker &>/dev/null; then
        echo "❌  docker not found. Please install Docker." >&2
        exit 1
    fi

    echo "▶  Building images and starting services …"
    docker compose up --build --abort-on-container-exit experiment

    echo ""
    echo "✅  Experiment finished."
    echo "    MLflow UI:  http://localhost:5001"
    echo ""
    echo "    To keep the MLflow UI running:"
    echo "      docker compose up mlflow"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODE — runs Python scripts directly
# ─────────────────────────────────────────────────────────────────────────────
echo "▶  Local mode selected."
echo "   DATA_DIR    = $DATA_DIR"
echo "   OUTPUT_DIR  = $OUTPUT_DIR"
echo "   MLFLOW_URI  = $MLFLOW_URI"
echo "   EXPERIMENT  = $EXPERIMENT_NAME"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "❌  python not found. Please install Python 3.10+ and activate your venv." >&2
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)
echo "▶  Using Python: $($PYTHON --version)"
echo ""

# ── Check data files exist ────────────────────────────────────────────────────
TRAIN_FILE="$DATA_DIR/train.parquet.snappy"
TEST_FILE="$DATA_DIR/test.parquet.snappy"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌  Train file not found: $TRAIN_FILE" >&2
    echo "    Place the data files inside $DATA_DIR/ or set DATA_DIR." >&2
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "⚠   Test file not found: $TEST_FILE — continuing without it."
fi

# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
echo "━━━  Step 1 / 2 — Preprocessing  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "$OUTPUT_DIR"

$PYTHON src/preprocess.py \
    --train-path "$TRAIN_FILE" \
    --test-path  "$TEST_FILE"  \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✅  Preprocessing complete."
echo "    Output: $OUTPUT_DIR/train_preprocessed.csv"
echo ""

# ── Step 2: Training with MLflow ──────────────────────────────────────────────
echo "━━━  Step 2 / 2 — Training with MLflow  ━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "    Tracking URI: $MLFLOW_URI"
echo "    Experiment  : $EXPERIMENT_NAME"
echo ""

MLFLOW_TRACKING_URI="$MLFLOW_URI" \
$PYTHON src/train.py \
    --data-dir    "$OUTPUT_DIR"       \
    --mlflow-uri  "$MLFLOW_URI"       \
    --experiment  "$EXPERIMENT_NAME"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Experiment complete!"
echo ""
echo "   View results in MLflow UI:"
echo "     → $MLFLOW_URI"
echo ""
echo "   Or start the MLflow UI locally:"
echo "     mlflow ui --port 5000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
