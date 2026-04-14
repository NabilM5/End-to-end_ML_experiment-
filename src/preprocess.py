"""
preprocess.py — Data preprocessing script for the e-commerce product categorisation project.

Usage:
    python src/preprocess.py \
        --train-path data/train.parquet.snappy \
        --test-path  data/test.parquet.snappy  \
        --output-dir output/

The script:
  1. Loads train / test parquet files.
  2. Fills missing values in text columns.
  3. Combines all text columns into a single 'combined_text' feature.
  4. Saves cleaned CSVs to --output-dir.

Adapted from hw4/kaggle_text_model.py (text-feature engineering).
"""

import argparse
import logging
import os

import pandas as pd

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
# Text columns present in the hw4 e-commerce dataset.
TEXT_COLS = ["name", "description", "model", "type_prefix", "vendor", "url"]

TARGET_COL = "category_ind"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    """Load a (snappy-compressed) parquet file and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    log.info("Loading %s …", path)
    df = pd.read_parquet(path)
    log.info("  → shape: %s", df.shape)
    return df


def combine_text(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Concatenate text columns into one whitespace-separated string per row.

    Columns missing from the DataFrame are silently skipped so that the same
    function works for both train (has TARGET_COL) and test data.
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        raise ValueError(f"None of {cols} found in DataFrame columns: {df.columns.tolist()}")

    parts = [df[c].fillna("").astype(str) for c in available]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " " + part
    return combined.str.strip()


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean a raw DataFrame:
      - Report missing-value counts.
      - Fill missing numeric columns with median.
      - Fill missing text columns with empty string.
      - Add 'combined_text' feature.
      - For train data, cast target to int.

    Returns a new DataFrame with the original columns plus 'combined_text'.
    """
    df = df.copy()

    # ── missing value report ────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        log.info("Missing values detected:")
        for col, cnt in missing.items():
            pct = 100 * cnt / len(df)
            log.info("  %-30s %6d  (%.1f%%)", col, cnt, pct)
    else:
        log.info("No missing values found.")

    # ── impute text columns ─────────────────────────────────────────────────
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ── impute numeric columns with median ──────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if is_train and TARGET_COL in num_cols:
        # Don't impute the target
        num_cols = [c for c in num_cols if c != TARGET_COL]
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info("  Imputed numeric column '%s' with median=%.4f", col, median_val)

    # ── add combined text feature ───────────────────────────────────────────
    df["combined_text"] = combine_text(df, TEXT_COLS)
    log.info("Combined text feature created from columns: %s", TEXT_COLS)

    # ── cast target ─────────────────────────────────────────────────────────
    if is_train and TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        log.info("Target '%s' cast to int. Unique classes: %d", TARGET_COL, df[TARGET_COL].nunique())

    return df


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess e-commerce product data for category classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        default="data/train.parquet.snappy",
        help="Path to the raw training parquet file.",
    )
    parser.add_argument(
        "--test-path",
        default="data/test.parquet.snappy",
        help="Path to the raw test parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where preprocessed CSVs will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── train ────────────────────────────────────────────────────────────────
    train_df = load_parquet(args.train_path)
    train_clean = clean_dataframe(train_df, is_train=True)
    train_out = os.path.join(args.output_dir, "train_preprocessed.csv")
    train_clean.to_csv(train_out, index=False)
    log.info("Saved preprocessed train → %s  (rows=%d)", train_out, len(train_clean))

    # ── test ─────────────────────────────────────────────────────────────────
    test_df = load_parquet(args.test_path)
    test_clean = clean_dataframe(test_df, is_train=False)
    test_out = os.path.join(args.output_dir, "test_preprocessed.csv")
    test_clean.to_csv(test_out, index=False)
    log.info("Saved preprocessed test  → %s  (rows=%d)", test_out, len(test_clean))

    log.info("Preprocessing complete ✓")


if __name__ == "__main__":
    main()
