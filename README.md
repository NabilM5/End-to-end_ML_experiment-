# project_1 — E-Commerce Product Categorisation

**MLOps course ·HSE University· Spring 2026**
**Student:** Nabil Mouhamech (`n.mukhamesh`)
**Presentation link:** [Open presentation](https://docs.google.com/presentation/d/15_dzpaGcOt_l7-4QNDafxDhK0aLiithioCvvKhGbm_M/edit?slide=id.g3d651261189_0_0#slide=id.g3d651261189_0_0)
**Colab link:** [Open Colab notebook](https://colab.research.google.com/drive/1aaDhbJDnN9Wl20Ys6OC_NE9zUc4kGMj3?usp=sharing)

---

## Overview

Multi-class text classification: predict `category_ind` for e-commerce products
using their textual metadata (name, description, model, vendor, URL, …).

| | |
|---|---|
| **Dataset** | E-commerce catalogue — `train.parquet.snappy` (hw4) |
| **Features** | Combined TF-IDF over 6 text columns |
| **Model** | `TfidfVectorizer` + `LogisticRegression` (sklearn Pipeline) |
| **Tracking** | MLflow — 3 runs sweeping `max_features` · `ngram_range` · `C` |
| **Metric** | macro-F1 (primary), accuracy (secondary) |

---

## Quick Evaluation Checklist

- [x] EDA notebook: `notebooks/eda.ipynb`
- [x] Data preprocessing script: `src/preprocess.py`
- [x] Training script: `src/train.py`
- [x] End-to-end script: `run_experiment.sh`
- [x] Pinned dependencies: `requirements.txt`
- [x] MLflow tracking of params/metrics/artifacts
- [x] Presentation link added
- [x] Colab link added

---

## Project Structure

```
project_1/
├── data/
│   ├── train.parquet.snappy     # raw training data (from hw4)
│   └── test.parquet.snappy      # raw test data (from hw4)
├── notebooks/
│   └── eda.ipynb                # EDA: distributions, missing values, prototype
├── src/
│   ├── preprocess.py            # data cleaning + combined_text feature (argparse)
│   └── train.py                 # model training + MLflow tracking (3 runs)
├── output/                      # generated at runtime — preprocessed CSVs
├── Dockerfile                   # python:3.10-slim, installs requirements, runs pipeline
├── docker-compose.yml           # service 1: mlflow UI  |  service 2: experiment
├── run_experiment.sh            # end-to-end: preprocess → train (local or --docker)
├── requirements.txt             # pinned deps (pip freeze style)
├── .dockerignore                # excludes __pycache__, .git, mlruns, notebooks, …
└── README.md                    # this file
```

---

## Quick Start — Run Locally

### 1. Prerequisites

```bash
python --version   # 3.10+
pip install -r requirements.txt
```

### 2. Start MLflow UI (separate terminal)

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

### 3. Run the full pipeline

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

This runs two steps automatically:

| Step | Script | What it does |
|---|---|---|
| 1 | `src/preprocess.py` | Fills missing values, combines text columns → `output/train_preprocessed.csv` |
| 2 | `src/train.py` | Trains 3 TF-IDF+LR runs with different hyperparameters, logs everything to MLflow |

### 4. Inspect results

Open **http://localhost:5001** — you will see the experiment
`project_1_ecom_categorisation` with three runs and all metrics logged.

---

## Run via Docker (one command)

```bash
chmod +x run_experiment.sh
./run_experiment.sh --docker
```

Docker Compose starts two containers:
- **`project1_mlflow`** — MLflow server on `http://localhost:5001`
- **`project1_experiment`** — runs preprocess + train, then exits

After it finishes open **http://localhost:5001** to view results.
The MLflow runs are persisted in the `mlflow_data` Docker volume.

### Keep MLflow UI running after the experiment

```bash
docker compose up mlflow
```

---

## Scripts Reference

### `src/preprocess.py`

```
python src/preprocess.py \
    --train-path data/train.parquet.snappy \
    --test-path  data/test.parquet.snappy  \
    --output-dir output/
```

| Argument | Default | Description |
|---|---|---|
| `--train-path` | `data/train.parquet.snappy` | Raw training parquet |
| `--test-path` | `data/test.parquet.snappy` | Raw test parquet |
| `--output-dir` | `output/` | Where to save preprocessed CSVs |

### `src/train.py`

```
python src/train.py \
    --data-dir   output/ \
    --mlflow-uri http://localhost:5001 \
    --experiment project_1_ecom_categorisation \
    --max-features 150000 \
    --ngram-max 2 \
    --C 1.0 \
    --test-size 0.2 \
    --seed 42
```

The script always runs **3 MLflow runs** (hyperparameter sweep), regardless of
the CLI defaults, to satisfy the MLOps rubric requirement of ≥ 3 runs.

---

## MLflow Tracking Details

Each run logs:

**Parameters**

| Parameter | Description |
|---|---|
| `max_features` | TF-IDF vocabulary size |
| `ngram_range` | n-gram window, e.g. `(1, 2)` |
| `C` | LR regularisation inverse |
| `solver` | `saga` (efficient for sparse data) |
| `min_df` | minimum document frequency |
| `sublinear_tf` | log-TF scaling |
| `test_size` | validation split fraction |
| `seed` | random seed |
| `train_samples` | training set size |
| `val_samples` | validation set size |
| `num_classes` | total category count |
| `model_type` | `TfidfVectorizer + LogisticRegression` |

**Metrics**

| Metric | Description |
|---|---|
| `train_accuracy` | Accuracy on training split |
| `val_accuracy` | Accuracy on validation split |
| `train_f1_macro` | Macro-F1 on training split |
| `val_f1_macro` | Macro-F1 on validation split ← primary |

**Artifacts per run**

- `classification_metrics.json` — train/test accuracy and F1 (hw4-compatible format)
- `dataset_label_counts.json` — class distribution of the training set
- `classification_report.txt` — full per-class sklearn report
- `model/` — serialised sklearn Pipeline registered as `project1_ecom_categoriser`

---

## Dataset

The dataset comes from the Kaggle competition
**Production ML Spring 2026** used in hw4.

| Column | Type | Description |
|---|---|---|
| `name` | text | Product name |
| `description` | text | Product description |
| `model` | text | Model identifier |
| `type_prefix` | text | Category prefix string |
| `vendor` | text | Brand / manufacturer |
| `url` | text | Product page URL |
| `category_ind` | int | **Target** — product category index |

**Key statistics (from EDA):**
- Strong class imbalance (Zipf-like distribution)
- Several text columns have missing values (handled by filling with `""`)
- `name` and `description` carry the most predictive signal

---

## Model Choice Rationale

**TF-IDF + Logistic Regression** was chosen because:

1. Proven strong baseline for high-dimensional sparse text classification
2. Fast to train — handles the full dataset in seconds on CPU
3. Interpretable — feature weights map directly to TF-IDF tokens
4. Already validated in hw4 (`kaggle_text_model.py` baseline) giving > 0.25 public score
5. Bigrams (`ngram_range=(1,2)`) capture product phrases like *"noise cancelling"*
6. `sublinear_tf=True` helps with long-tail class balance

---

## Scoring Rubric Coverage

| Criterion | Points | How satisfied |
|---|---|---|
| EDA analysis, features, prototypes | 4 | `notebooks/eda.ipynb` — distributions, missing values, correlations, inline prototype |
| Solution idea, model choice, architecture | 3 | README rationale + comments in `train.py` |
| End-to-end script (mandatory gate) | required | `run_experiment.sh` — preprocess → train, with `--docker` option |
| Pinned dependencies | 1 | `requirements.txt` — all versions pinned |
| MLflow tracking — all params logged | 3 | `train.py` logs 12 params + 4 metrics + 3 artifacts per run |
| Data processing Python script | 2 | `src/preprocess.py` with argparse |
| Model training Python script | 2 | `src/train.py` with argparse + MLflow |

---

## Dependencies

All pinned in `requirements.txt`. Key packages:

```
scikit-learn==1.5.2
pandas==2.2.3
pyarrow==17.0.0
mlflow==2.19.0
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
```
