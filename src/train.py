"""
train.py — Model training script with MLflow tracking for the
            e-commerce product categorisation project.

Usage:
    python src/train.py \
        --data-dir      output/ \
        --mlflow-uri    http://localhost:5000 \
        --experiment    project_1_ecom_categorisation \
        --max-features  150000 \
        --ngram-max     2 \
        --C             1.0 \
        --test-size     0.2 \
        --seed          42

The script runs three MLflow experiment runs sweeping over (C, max_features)
hyperparameters so you can compare them in the MLflow UI.

Adapted from hw4/run_mlflow.py and hw4/kaggle_text_model.py.
"""

import argparse
import json
import logging
import os
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TARGET_COL = "category_ind"
TEXT_FEATURE = "combined_text"


# ── data loading ───────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """
    Load preprocessed CSVs produced by preprocess.py.

    Returns:
        train_df, test_df as DataFrames.
    """
    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path  = os.path.join(data_dir, "test_preprocessed.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Preprocessed train file not found: {train_path}. "
            "Run preprocess.py first."
        )

    log.info("Loading preprocessed train: %s", train_path)
    train_df = pd.read_csv(train_path)
    log.info("  → shape: %s", train_df.shape)

    test_df = None
    if os.path.exists(test_path):
        log.info("Loading preprocessed test: %s", test_path)
        test_df = pd.read_csv(test_path)
        log.info("  → shape: %s", test_df.shape)
    else:
        log.warning("No test file found at %s — skipping test predictions.", test_path)

    return train_df, test_df


# ── model building ─────────────────────────────────────────────────────────────

def build_pipeline(
    max_features: int,
    ngram_max: int,
    C: float,
    seed: int,
) -> Pipeline:
    """
    Build a sklearn Pipeline:
      TF-IDF (unigrams + ngrams)  →  Logistic Regression

    Args:
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_max:    Upper bound of n-gram range (1 = unigrams only, 2 = bigrams).
        C:            Inverse regularisation strength for LogisticRegression.
        seed:         Random seed for reproducibility.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=2,
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,   # Apply log(1 + tf) scaling — helps with skewed data
    )
    classifier = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="saga",
        multi_class="auto",
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# ── artefact helpers ───────────────────────────────────────────────────────────

def save_json_artifact(payload: dict, filename: str) -> str:
    """Write payload as JSON to a temp file and return the path."""
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def log_label_counts(train_df: pd.DataFrame):
    """Log class distribution of the target column as a JSON artifact."""
    counts = (
        train_df[TARGET_COL]
        .value_counts()
        .sort_index()
    )
    payload = {str(k): int(v) for k, v in counts.items()}
    path = save_json_artifact(payload, "dataset_label_counts.json")
    mlflow.log_artifact(path)
    os.remove(path)
    log.info("Logged dataset_label_counts.json (%d classes)", len(payload))


def log_metrics_artifact(metrics: dict):
    """Save classification metrics as a JSON artifact."""
    path = save_json_artifact(metrics, "classification_metrics.json")
    mlflow.log_artifact(path)
    os.remove(path)
    log.info("Logged classification_metrics.json")


# ── single experiment run ──────────────────────────────────────────────────────

def run_experiment(
    train_df: pd.DataFrame,
    *,
    max_features: int,
    ngram_max: int,
    C: float,
    test_size: float,
    seed: int,
    run_name: str,
):
    """
    Perform one MLflow run:
      1. Split data into train / validation sets.
      2. Fit the TF-IDF + LogisticRegression pipeline.
      3. Compute accuracy and macro-F1 on both splits.
      4. Log all parameters, metrics, artefacts and the model.
    """
    # ── split ───────────────────────────────────────────────────────────────
    X = train_df[TEXT_FEATURE].astype(str)
    y = train_df[TARGET_COL].astype(int)

    min_count = y.value_counts().min()
    stratify = y if min_count >= 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    log.info(
        "Train size: %d  |  Validation size: %d",
        len(X_train), len(X_val),
    )

    # ── build & fit model ───────────────────────────────────────────────────
    pipeline = build_pipeline(
        max_features=max_features,
        ngram_max=ngram_max,
        C=C,
        seed=seed,
    )

    with mlflow.start_run(run_name=run_name):

        # ── log all hyper-parameters ──────────────────────────────────────
        mlflow.log_params({
            "max_features":   max_features,
            "ngram_range":    f"(1, {ngram_max})",
            "C":              C,
            "solver":         "saga",
            "min_df":         2,
            "sublinear_tf":   True,
            "test_size":      test_size,
            "seed":           seed,
            "model_type":     "TfidfVectorizer + LogisticRegression",
            "train_samples":  len(X_train),
            "val_samples":    len(X_val),
            "num_classes":    y.nunique(),
        })

        log.info("Fitting pipeline (max_features=%d, ngram_max=%d, C=%.3f) …",
                 max_features, ngram_max, C)
        pipeline.fit(X_train, y_train)

        # ── predictions ───────────────────────────────────────────────────
        y_pred_train = pipeline.predict(X_train)
        y_pred_val   = pipeline.predict(X_val)

        # ── compute metrics ───────────────────────────────────────────────
        train_accuracy  = accuracy_score(y_train, y_pred_train)
        val_accuracy    = accuracy_score(y_val,   y_pred_val)
        train_f1_macro  = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
        val_f1_macro    = f1_score(y_val,   y_pred_val,   average="macro", zero_division=0)

        # ── log all metrics to MLflow UI ──────────────────────────────────
        mlflow.log_metrics({
            "train_accuracy":  round(train_accuracy, 4),
            "val_accuracy":    round(val_accuracy,   4),
            "train_f1_macro":  round(train_f1_macro, 4),
            "val_f1_macro":    round(val_f1_macro,   4),
        })

        log.info(
            "Metrics → train_acc=%.4f  val_acc=%.4f  "
            "train_f1=%.4f  val_f1=%.4f",
            train_accuracy, val_accuracy,
            train_f1_macro, val_f1_macro,
        )

        # ── JSON artefacts (compatible with hw4 grading format) ───────────
        metrics_payload = {
            "train_accuracy":  round(train_accuracy, 4),
            "test_accuracy":   round(val_accuracy,   4),   # val treated as test
            "train_f1_macro":  round(train_f1_macro, 4),
            "test_f1_macro":   round(val_f1_macro,   4),
        }
        log_metrics_artifact(metrics_payload)
        log_label_counts(train_df)

        # ── log classification report as text artifact ────────────────────
        report = classification_report(y_val, y_pred_val, zero_division=0)
        report_path = os.path.join(tempfile.gettempdir(), "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # ── log the trained model ─────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="project1_ecom_categoriser",
        )
        log.info("Model logged to MLflow ✓")

    return {
        "run_name":       run_name,
        "val_accuracy":   val_accuracy,
        "val_f1_macro":   val_f1_macro,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + LR model with MLflow tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",    default="output",
                        help="Directory with preprocessed CSVs (output of preprocess.py).")
    parser.add_argument("--mlflow-uri",  default="http://localhost:5000",
                        help="MLflow tracking server URI.")
    parser.add_argument("--experiment",  default="project_1_ecom_categorisation",
                        help="MLflow experiment name.")
    parser.add_argument("--max-features", type=int,   default=150_000,
                        help="TF-IDF max vocabulary size.")
    parser.add_argument("--ngram-max",    type=int,   default=2,
                        help="Max n-gram size (1=unigrams, 2=bigrams).")
    parser.add_argument("--C",            type=float, default=1.0,
                        help="Logistic Regression regularisation inverse (C).")
    parser.add_argument("--test-size",    type=float, default=0.2,
                        help="Fraction of data held out for validation.")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    log.info("MLflow tracking URI : %s", args.mlflow_uri)
    log.info("MLflow experiment   : %s", args.experiment)

    # ── load data ─────────────────────────────────────────────────────────────
    train_df, _ = load_data(args.data_dir)

    # ── hyperparameter grid (3 runs as required by MLOps rubric) ──────────────
    sweep = [
        # run 1 — baseline (matches script default)
        dict(max_features=100_000, ngram_max=1, C=1.0,  seed=args.seed),
        # run 2 — bigrams, larger vocabulary
        dict(max_features=150_000, ngram_max=2, C=1.0,  seed=args.seed),
        # run 3 — stronger regularisation
        dict(max_features=150_000, ngram_max=2, C=5.0,  seed=args.seed),
    ]

    results = []
    for i, cfg in enumerate(sweep, start=1):
        run_name = (
            f"run_{i}_feat{cfg['max_features']//1000}k_"
            f"ng{cfg['ngram_max']}_C{cfg['C']}"
        )
        log.info("━━━ Starting run %d / %d : %s ━━━", i, len(sweep), run_name)
        result = run_experiment(
            train_df,
            run_name=run_name,
            test_size=args.test_size,
            **cfg,
        )
        results.append(result)

    # ── summary ──────────────────────────────────────────────────────────────
    log.info("\n%s", "=" * 60)
    log.info("Run summary:")
    best = max(results, key=lambda r: r["val_f1_macro"])
    for r in results:
        marker = " ← best" if r["run_name"] == best["run_name"] else ""
        log.info(
            "  %-55s  val_acc=%.4f  val_f1=%.4f%s",
            r["run_name"], r["val_accuracy"], r["val_f1_macro"], marker,
        )
    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
