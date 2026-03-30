# Problem Day Playbook

This repo is intentionally split into:

- invariant infrastructure we can finish before the problem statement
- domain-specific logic we swap once the task is revealed

The goal is to make problem-day work mostly a focused domain integration, not a full-stack rebuild.

## What Is Already Stable

These pieces should stay mostly unchanged across domains:

- [app.py](/Users/ariyanbhakat/Desktop/metax/app.py): FastAPI API surface, HTTP endpoints, WebSocket endpoint, session handling
- [Dockerfile](/Users/ariyanbhakat/Desktop/metax/Dockerfile): container runtime and HF Spaces packaging
- [inference.py](/Users/ariyanbhakat/Desktop/metax/inference.py): root inference entrypoint using `OpenAI(...)`
- [scripts/validate.py](/Users/ariyanbhakat/Desktop/metax/scripts/validate.py): submission validation checks
- [tests/test_api.py](/Users/ariyanbhakat/Desktop/metax/tests/test_api.py): API contract coverage
- [openenv.yaml](/Users/ariyanbhakat/Desktop/metax/openenv.yaml): metadata shell and endpoint manifest

## What Must Change When The Problem Drops

These are the main swap surfaces:

- [environment/tasks.py](/Users/ariyanbhakat/Desktop/metax/environment/tasks.py)
- [environment/graders.py](/Users/ariyanbhakat/Desktop/metax/environment/graders.py)
- [environment/reward.py](/Users/ariyanbhakat/Desktop/metax/environment/reward.py)
- [environment/models.py](/Users/ariyanbhakat/Desktop/metax/environment/models.py) if the action or observation schema changes materially
- [README.md](/Users/ariyanbhakat/Desktop/metax/README.md)
- [openenv.yaml](/Users/ariyanbhakat/Desktop/metax/openenv.yaml)
- [inference.py](/Users/ariyanbhakat/Desktop/metax/inference.py) only if the prompting/action parsing needs domain-specific behavior

## Core OpenEnv Principles To Preserve

Based on the OpenEnv docs you shared, the final environment should keep these properties:

- Type-safe `Observation`, `Action`, and `Reward` models
- Clean `reset()`, `step(action)`, and `state()` environment lifecycle
- Container-first deployment with reproducible runtime behavior
- Clear client/server separation
- Session isolation per HTTP session or WebSocket connection
- Minimal assumptions in the transport layer so the domain logic stays swappable

## Problem-Day Sequence

1. Read the official task statement carefully and identify:
   - the real-world task
   - the unit of work
   - what the agent can observe
   - what the agent can do
   - what success looks like
2. Rewrite the domain schema in [environment/models.py](/Users/ariyanbhakat/Desktop/metax/environment/models.py):
   - observation fields
   - action parameters
   - any domain metadata
3. Replace the task registry in [environment/tasks.py](/Users/ariyanbhakat/Desktop/metax/environment/tasks.py):
   - add at least 3 tasks
   - easy, medium, hard progression
   - realistic descriptions
4. Implement deterministic graders in [environment/graders.py](/Users/ariyanbhakat/Desktop/metax/environment/graders.py):
   - score range `[0.0, 1.0]`
   - no randomness
   - partial credit where appropriate
5. Rewrite reward shaping in [environment/reward.py](/Users/ariyanbhakat/Desktop/metax/environment/reward.py):
   - progress signals every step
   - penalties for wasteful or invalid actions
   - keep normalization bounded
6. Update prompting or fallback behavior in [inference.py](/Users/ariyanbhakat/Desktop/metax/inference.py)
7. Update metadata and docs:
   - [openenv.yaml](/Users/ariyanbhakat/Desktop/metax/openenv.yaml)
   - [README.md](/Users/ariyanbhakat/Desktop/metax/README.md)
8. Run final checks:
   - `python -m pytest -q`
   - `python scripts/validate.py`

## Domain Design Checklist

Before we call a domain “done”, it should pass this smell test:

- Is this something a human actually does?
- Are the observations realistic enough for an agent to reason over?
- Are the actions explicit and constrained?
- Do the easy, medium, and hard tasks differ meaningfully?
- Does the grader reward real progress rather than a single shortcut?
- Could a generic strong LLM attempt this sensibly from the observation?

## Submission Gates

We should assume the validator will care about all of the following:

- the Space deploys and responds
- `/health` returns `200`
- `/reset`, `/step`, `/state`, `/tasks`, `/grader` behave correctly
- `openenv.yaml` is present and coherent
- the Dockerfile builds
- root [inference.py](/Users/ariyanbhakat/Desktop/metax/inference.py) exists and runs without crashing
- at least 3 tasks exist
- graders return non-constant, bounded scores

## Nice-To-Have But Not Blockers

- richer WebSocket client examples
- stronger baseline prompting
- CI
- benchmark scripts
- sample notebooks

These help polish, but the core win on problem day is a strong domain layer plus reliable validation.

## Judge Optimization

Use [PROBLEM_DAY_SCORECARD.md](/Users/ariyanbhakat/Desktop/metax/PROBLEM_DAY_SCORECARD.md) to compare candidate domain interpretations and to sanity-check the final build against the actual judging weights:

- real-world utility: `30%`
- task & grader quality: `25%`
- environment design: `20%`
- code quality & spec compliance: `15%`
- creativity & novelty: `10%`
