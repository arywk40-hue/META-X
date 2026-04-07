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

This repository is a production-ready OpenEnv environment for real-world tabular data cleaning. The core submission is an RL benchmark where agents inspect dirty tables, take structured cleaning actions, receive dense rewards and feedback, and are scored by deterministic graders across fixed easy/medium/hard tasks.

Around that benchmark, the repo also includes an optional dataset-preparation studio for arbitrary CSV files. That extension is useful for demos and real datasets, but the primary submission surface remains the OpenEnv environment itself: tasks, actions, rewards, graders, and evaluation.

## OpenEnv Core

- `reset()` starts a fresh benchmark episode with a clean task state.
- `step(action)` applies one structured cleaning action and returns observation, reward, done, and info.
- `state()` exposes the current episode state, history, and score context.
- typed `Action`, `Observation`, `Reward`, and request models live in [environment/models.py](environment/models.py)
- the core episode engine lives in [environment/env.py](environment/env.py)
- fixed benchmark tasks and deterministic graders live in [environment/tasks.py](environment/tasks.py) and [environment/graders.py](environment/graders.py)

## Benchmark First

The benchmark is the main submission target:

- six fixed data-cleaning tasks across easy, medium, and hard difficulty
- deterministic graders with bounded scores in `[0.0, 1.0]`
- dense reward shaping with partial progress and penalties
- stable `/tasks` output for validation and judging

The dynamic CSV workflow and Streamlit interface are secondary layers built around the benchmark, not replacements for it.

Determinism note:

- the fixed benchmark tasks and graders do not depend on an LLM
- the optional LLM planner/reviewer loop is limited to the studio workflow
- even there, only validated transformations are allowed to touch the dataframe

## What Makes This Different

- It models a real data-cleaning workflow instead of a toy control task.
- It combines a stable benchmark with six fixed OpenEnv tasks and a dynamic CSV-to-task bridge for unseen datasets.
- EDA suggestions are not only displayed: validated transformations are actually executed and exported into train-ready artifacts.
- The same repo supports benchmarking, arbitrary CSV cleaning, feature engineering, and fast baseline evaluation in one place, while keeping the OpenEnv benchmark surface clean.

## Modes

The project has three clear operating modes:

- `Benchmark Mode`: the fixed OpenEnv benchmark exposed through `/tasks`, with deterministic graders and dense rewards.
- `Dynamic Task Mode`: any uploaded CSV can be converted into a session-local RL cleaning episode without changing the public benchmark.
- `Dataset Studio Mode`: an optional demo/workflow layer where arbitrary CSVs can be profiled, feature engineered, prepared, and evaluated into train-ready outputs.

## Architecture At A Glance

- [app.py](app.py): FastAPI/OpenEnv HTTP surface, session management, and dataset prep/eval endpoints.
- [environment/env.py](environment/env.py): the core `reset()/step()/state()` episode engine.
- [environment/tasks.py](environment/tasks.py) and [environment/graders.py](environment/graders.py): fixed benchmark definitions and deterministic scoring.
- [environment/dynamic_task_generator.py](environment/dynamic_task_generator.py): arbitrary CSV -> session-local OpenEnv task generation around the fixed benchmark.
- [environment/eda_agent.py](environment/eda_agent.py): schema-grounded EDA, validated feature engineering, and planner/reviewer orchestration.
- [environment/data_prep.py](environment/data_prep.py) and [environment/evaluation.py](environment/evaluation.py): final dataset preparation, artifact export, and fast baseline model evaluation.
- [streamlit_app.py](streamlit_app.py): demo UI for arbitrary tabular CSV workflows layered on top of the environment.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000` locally and `7860` in Hugging Face Spaces.

## One-Minute Benchmark Demo

Run the core benchmark path locally:

```bash
uvicorn app:app --reload
python inference.py
```

That starts the OpenEnv server, runs the submission baseline against the fixed benchmark, and emits the required `[START] / [STEP] / [END]` logs.

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

### Repository Layout

This repo is intentionally organized so judges can find the submission-critical pieces first:

```text
metax/
├── README.md                  # submission overview and usage
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # HF Spaces / container entrypoint
├── inference.py               # submission baseline script
├── app.py                     # FastAPI/OpenEnv HTTP surface
├── streamlit_app.py           # demo UI for arbitrary CSV datasets
├── prepare_dataset.py         # CLI: prepare any CSV
├── prepare_and_evaluate.py    # CLI: prepare + score any CSV
├── environment/
│   ├── models.py              # typed OpenEnv + dataset request models
│   ├── env.py                 # reset()/step()/state() implementation
│   ├── tasks.py               # fixed benchmark tasks
│   ├── graders.py             # deterministic graders
│   ├── reward.py              # dense reward shaping
│   ├── dynamic_task_generator.py
│   ├── eda_agent.py
│   ├── data_prep.py
│   ├── evaluation.py
│   └── reporting.py
├── baseline/
│   └── inference.py           # internal heuristic smoke baseline
├── scripts/
│   ├── validate.py
│   └── validate-submission.sh
├── tests/
│   └── ...
└── data/
    └── kaggle/
```

### Submission Surface

If a reviewer only reads a few files, start here:

- [openenv.yaml](openenv.yaml)
- [inference.py](inference.py)
- [app.py](app.py)
- [environment/env.py](environment/env.py)
- [environment/tasks.py](environment/tasks.py)
- [environment/graders.py](environment/graders.py)

## Benchmark

The environment now ships with six data-cleaning tasks across three difficulty tiers:

- Easy: null repair
- Medium: date and currency standardization, duplicate and outlier cleanup, Titanic manifest cleanup
- Hard: multi-layer pipeline repair, adversarial sensor validation

Only public task metadata is exposed via `/tasks`. The graders score action histories deterministically and reward partial progress.

In addition to the fixed benchmark, the app can generate a **session-local dynamic cleaning episode** from any uploaded CSV. These dynamic tasks do not appear in `/tasks`, so the benchmark stays stable for validation and judging.

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

The submission-compatible inference entrypoint lives at the repo root as `inference.py`. It uses the OpenAI client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, then drives the OpenEnv HTTP API.

Start the environment first:

```bash
uvicorn app:app --reload
```

Then run inference:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=your-model-name
export HF_TOKEN=your_api_key
python inference.py
```

You can still run the internal baseline module with:

```bash
python -m baseline.inference
```

The internal baseline keeps a heuristic fallback for smoke tests and CI. The root `inference.py` is the submission-oriented script.

Required submission environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Local developer convenience fallbacks still work for compatible providers, but the submission path is built around `HF_TOKEN`.

If you are using an OpenAI-compatible provider such as Groq, the safest submission configuration is:

- keep using the `OpenAI(...)` client in [inference.py](inference.py)
- set `API_BASE_URL` to your provider's OpenAI-compatible endpoint
- set `MODEL_NAME` to the provider model name
- put the active provider key into `HF_TOKEN` for submission-time compatibility

### Required STDOUT Format

The root `inference.py` now emits only the evaluator-facing line types on stdout:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

Notes:

- one `[START]` line per episode
- one `[STEP]` line immediately after each environment step
- one `[END]` line always emitted, even if inference falls back or errors
- all rewards are formatted to 2 decimals
- `success` and `done` are lowercase booleans
- any unexpected diagnostic output is sent to stderr, not stdout

### Submission Checklist

Before submitting, run the following:

```bash
python -m pytest -q
python scripts/validate.py
openenv validate
docker build .
python inference.py
```

For hosted validation, also confirm:

- your HF Space returns `200` on `POST /reset`
- the submitted `inference.py` completes without error
- the root script stays under the `2 vCPU / 8 GB / <20 min` constraints

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
  --output-dir outputs/titanic \
  --use-eda-agent
```

This writes:

- `*_prepared_full.csv`
- `*_prepared_train.csv`
- `*_prepared_valid.csv`
- `*_feature_manifest.json`
- `reports/*_profile.json`
- `reports/*_work_queue.json`
- `reports/*_report.md`
- `reports/*_report.tex`
- `reports/graphs/*.svg`
- `reports/*_eda_report.json` and `reports/*_eda_report.md` when the EDA agent is enabled

There is also an HTTP endpoint:

```bash
curl -X POST http://127.0.0.1:8000/prepare-dataset \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_path": "data/kaggle/Titanic-Dataset.csv",
    "target_column": "Survived",
    "output_dir": "outputs/titanic",
    "use_eda_agent": true,
    "eda_use_llm": false
  }'
```

## EDA Agent

The repo now includes a standalone EDA agent in [eda_agent.py](eda_agent.py) and [environment/eda_agent.py](environment/eda_agent.py).

It does three useful things for arbitrary new datasets:

- profiles columns without assuming fixed schemas
- surfaces data-quality issues and correlation insights
- proposes feature-engineering steps that are validated against the live dataframe schema before they can be applied

Run it directly:

```bash
python eda_agent.py data/kaggle/Titanic-Dataset.csv Survived false
```

The last argument controls LLM use:

- `false` = deterministic statistical EDA only
- omit it or pass `true` = deterministic EDA plus validated LLM suggestions when credentials exist

The important behavior is:

- deterministic profiling always runs first
- heuristic feature-engineering steps are always available
- the LLM can only add extra ideas on top
- unsafe or schema-breaking LLM steps are rejected before they touch the dataset

## Prepare and Evaluate

To prepare a dataset and immediately score it with fast baseline models:

```bash
python prepare_and_evaluate.py \
  --csv data/kaggle/Titanic-Dataset.csv \
  --target Survived \
  --output-dir outputs/titanic_eval \
  --use-eda-agent
```

This writes the prepared CSVs plus an evaluation report that ranks candidate models on the validation split.

API version:

```bash
curl -X POST http://127.0.0.1:8000/prepare-and-evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_path": "data/kaggle/Titanic-Dataset.csv",
    "target_column": "Survived",
    "output_dir": "outputs/titanic_eval",
    "use_eda_agent": true,
    "eda_use_llm": false
  }'
```

## Dynamic OpenEnv Tasks From Any CSV

For arbitrary unseen schemas, you can generate a one-off OpenEnv cleaning episode directly from a CSV:

```bash
curl -X POST http://127.0.0.1:8000/generate-dynamic-task \
  -H 'Content-Type: application/json' \
  -d '{
    "csv_path": "data/kaggle/Titanic-Dataset.csv",
    "max_issues": 7,
    "max_preview_rows": 12
  }'
```

This endpoint:

- profiles the uploaded CSV deterministically
- detects real issues like missing values, literal null strings, duplicates, outliers, negative values, and date-format problems
- creates a **session-local** OpenEnv task plus matching deterministic grader
- returns a normal observation/session pair that you can continue with `/step`, `/state`, and `/grader`

The important design choice is that these dynamic tasks are **not added to the shared benchmark registry**. `/tasks` stays fixed and reproducible, while custom CSV episodes stay isolated to the current session.

## Streamlit UI

Launch the CSV upload UI:

```bash
streamlit run streamlit_app.py
```

The UI supports:

- CSV upload
- dataset preview and missing-value profile
- EDA-agent toggle for unknown or changing schemas
- dynamic OpenEnv task generation from the uploaded CSV
- target-column selection
- prepare-only mode
- prepare-and-evaluate mode
- direct download of train, validation, manifest, profile/work-queue JSON, Markdown/LaTeX reports, and evaluation artifacts

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

