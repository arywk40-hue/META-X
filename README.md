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

The environment now ships with six data-cleaning tasks across three difficulty tiers:

- Easy: null repair
- Medium: date and currency standardization, duplicate and outlier cleanup, Titanic manifest cleanup
- Hard: multi-layer pipeline repair, adversarial sensor validation

Only public task metadata is exposed via `/tasks`. The graders score action histories deterministically and reward partial progress.

## Action, Observation, Reward

Action space:

- `action_type`: one of `fix_value`, `drop_row`, `fill_missing`, `cast_type`, `rename_column`, `flag_anomaly`, `standardize_format`
- `row_index`: 0-based row reference for row-level actions
- `column`: target column name
- `new_value`: replacement value when applicable
- `reason`: short justification for the change

Observation space:

- `task_id`, `task_name`, `task_description`
- `dataset_preview`: the current mutable table preview after the latest action
- `issues_remaining`
- `step`, `max_steps`, `attempts_remaining`
- `feedback`, `feedback_history`
- `available_actions`

Reward space:

- `value` in `[0.0, 1.0]`
- `issues_fixed_this_step`
- `issues_remaining`
- `solved`
- `attempts_used`

## Task Details

- `null_filling` (`easy`): repair missing and malformed nulls in a contacts table
- `format_standardization` (`medium`): normalize date and currency formats in a revenue table
- `duplicate_outlier` (`medium`): remove duplicates and handle obvious transaction anomalies
- `titanic_manifest` (`medium`): repair real missing values in a Titanic manifest slice
- `multi_layer_pipeline` (`hard`): resolve interacting foreign-key, date, duplicate, and pricing issues in an orders pipeline
- `adversarial_sensor` (`hard`): fix only true anomalies while avoiding false-positive edits to physically valid readings

## Inference

The hackathon-compatible inference entrypoint lives at the repo root as `inference.py`. It uses the OpenAI client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, then drives the OpenEnv HTTP API.

Start the environment first:

```bash
uvicorn app:app --reload
```

Then run inference:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=your-model-name
export OPENAI_API_KEY=your_api_key
python inference.py
```

You can still run the internal baseline module with:

```bash
python -m baseline.inference
```

The internal baseline keeps a heuristic fallback for smoke tests and CI. The root `inference.py` is the submission-oriented script.

The root script accepts:

- `OPENAI_API_KEY` as the primary API key variable
- `API_BASE_URL`
- `MODEL_NAME`
- optional `HF_TOKEN` or `GROQ_API_KEY` for compatible providers

## Baseline Scores

Reproducible heuristic baseline:

```bash
python -m baseline.inference --task-id null_filling --task-id format_standardization --task-id duplicate_outlier --task-id titanic_manifest --task-id multi_layer_pipeline --task-id adversarial_sensor
```

Observed heuristic baseline results on the current benchmark:

| Task | Difficulty | Score | Steps |
|---|---|---:|---:|
| `null_filling` | easy | 1.00 | 3 |
| `format_standardization` | medium | 1.00 | 5 |
| `duplicate_outlier` | medium | 1.00 | 4 |
| `titanic_manifest` | medium | 1.00 | 4 |
| `multi_layer_pipeline` | hard | 1.00 | 7 |
| `adversarial_sensor` | hard | 1.00 | 2 |

Summary:

- `total_episodes = 6`
- `total_steps = 25`
- `overall_mean = 1.00`
- `runner_mode = heuristic`

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

## Final Submission Check

This repo now passes:

- `python -m pytest -q`
- `python scripts/validate.py`
- `openenv validate`

To run the exact 3-step prevalidation flow before submission:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://arywk-40-code-review-env.hf.space .
```

That script checks:

1. live HF Space `/reset`
2. `docker build`
3. `openenv validate`

If `docker` is not installed on your machine, install Docker first and rerun the script.
