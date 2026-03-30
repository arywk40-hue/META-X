---
title: Code Review Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rl-environment
  - code-review
---

# OpenEnv RL Environment Template

This repository is a production-ready starter for the Meta PyTorch OpenEnv Hackathon. It ships with a full FastAPI environment, deterministic tasks and graders, dense reward shaping, a baseline agent runner, and a test suite that can be swapped to the final domain on problem day.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000` locally and `7860` in HuggingFace Spaces.

## Transport

The template supports both:

- HTTP endpoints for validator compatibility and simple integrations
- WebSocket sessions at `/ws` for OpenEnv-style per-connection environment instances

WebSocket messages use a simple envelope:

```json
{"type":"reset","payload":{"task_id":"easy_01"}}
{"type":"step","payload":{"action_type":"classify_a"}}
{"type":"state"}
{"type":"tasks"}
{"type":"grader","payload":{"task_id":"easy_01","episode":[...]}}
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

## Swap Strategy

The default domain is classification so the template is immediately runnable. When the hackathon problem drops, the primary swap surface is:

- `environment/tasks.py`
- `environment/graders.py`
- `environment/reward.py`
- `environment/models.py` if the observation/action payloads need new fields

For the exact launch-day sequence, see [PROBLEM_DAY_PLAYBOOK.md](/Users/ariyanbhakat/Desktop/metax/PROBLEM_DAY_PLAYBOOK.md).
For judging-weighted decision-making, use [PROBLEM_DAY_SCORECARD.md](/Users/ariyanbhakat/Desktop/metax/PROBLEM_DAY_SCORECARD.md).

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
