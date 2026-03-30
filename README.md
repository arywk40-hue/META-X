---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - rl-environment
  - data-cleaning
---

# OpenEnv Data Cleaning Environment

This repository is a production-ready OpenEnv environment for real-world tabular data cleaning. It exposes a FastAPI-compatible API, deterministic graders, dense reward shaping, and a root `inference.py` runner that uses an OpenAI-compatible client for live evaluation.

It also now includes a generic dataset-preparation workflow for arbitrary CSV files. You can feed in a dataset, optionally specify a target column, and get back train-ready CSV artifacts plus a feature manifest.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000` locally and `7860` in Hugging Face Spaces.

## Transport

The template supports both:

- HTTP endpoints for validator compatibility and simple integrations
- WebSocket sessions at `/ws` for OpenEnv-style per-connection environment instances

WebSocket messages use a simple envelope:

```json
{"type":"reset","payload":{"task_id":"null_filling"}}
{"type":"step","payload":{"action_type":"fill_missing","row_index":1,"column":"age","new_value":"unknown","reason":"age is missing"}}
{"type":"state"}
{"type":"tasks"}
{"type":"grader","payload":{"task_id":"null_filling","episode":[...]}}
```

Each WebSocket connection gets its own isolated `OpenEnv` instance and is cleaned up automatically when the connection closes.

## Runtime Scaling

The server now accepts basic runtime scaling knobs inspired by OpenEnv guidance:

- `HOST`
- `PORT`
- `WORKERS`
- `MAX_CONCURRENT_ENVS`

Example:

```bash
HOST=0.0.0.0 PORT=7860 WORKERS=2 MAX_CONCURRENT_ENVS=128 uvicorn app:app --host 0.0.0.0 --port 7860 --workers 2
```

## Project Layout

- `app.py`: FastAPI API surface for OpenEnv-compatible endpoints.
- `environment/`: environment engine, Pydantic models, tasks, graders, and rewards.
- `baseline/`: baseline runner with heuristic fallback and OpenAI integration.
- `scripts/validate.py`: smoke validation for metadata, graders, and endpoints.
- `tests/`: unit and integration coverage for the template.

## Benchmark

The environment now ships with five data-cleaning tasks across three difficulty tiers:

- Easy: null repair
- Medium: date and currency standardization, duplicate and outlier cleanup
- Hard: multi-layer pipeline repair, adversarial sensor validation

Only public task metadata is exposed via `/tasks`. The graders score action histories deterministically and reward partial progress.

## Inference

The hackathon-compatible inference entrypoint lives at the repo root as `inference.py`. It uses the OpenAI client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, then drives the OpenEnv HTTP API.

Start the environment first:

```bash
uvicorn app:app --reload
```

Then run inference:

```bash
python inference.py
```

You can still run the internal baseline module with:

```bash
python -m baseline.inference
```

The internal baseline keeps a heuristic fallback for smoke tests and CI. The root `inference.py` is the submission-oriented script.

## Generic CSV Preparation

Prepare any CSV into train-ready artifacts:

```bash
python prepare_dataset.py \
  --csv data/kaggle/Titanic-Dataset.csv \
  --target Survived \
  --output-dir outputs/titanic
```

This writes:

- `*_prepared_full.csv`
- `*_prepared_train.csv`
- `*_prepared_valid.csv`
- `*_feature_manifest.json`

There is also an HTTP endpoint:

```bash
curl -X POST http://127.0.0.1:8000/prepare-dataset \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_path": "data/kaggle/Titanic-Dataset.csv",
    "target_column": "Survived",
    "output_dir": "outputs/titanic"
  }'
```

## Prepare and Evaluate

To prepare a dataset and immediately score it with fast baseline models:

```bash
python prepare_and_evaluate.py \
  --csv data/kaggle/Titanic-Dataset.csv \
  --target Survived \
  --output-dir outputs/titanic_eval
```

This writes the prepared CSVs plus an evaluation report that ranks candidate models on the validation split.

API version:

```bash
curl -X POST http://127.0.0.1:8000/prepare-and-evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_path": "data/kaggle/Titanic-Dataset.csv",
    "target_column": "Survived",
    "output_dir": "outputs/titanic_eval"
  }'
```

## Streamlit UI

Launch the CSV upload UI:

```bash
streamlit run streamlit_app.py
```

The UI supports:

- CSV upload
- dataset preview and missing-value profile
- target-column selection
- prepare-only mode
- prepare-and-evaluate mode
- direct download of train, validation, manifest, and evaluation report artifacts
