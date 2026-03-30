# OpenEnv RL Environment Template

This repository implements the architecture in the design brief as a working scaffold:

- `app.py` exposes the required OpenEnv HTTP endpoints.
- `environment/` contains the stateful environment engine, typed models, default task registry, deterministic graders, and dense reward logic.
- `baseline/` contains a baseline runner that uses OpenAI when available and falls back to a deterministic heuristic for offline smoke tests.
- `scripts/validate.py` provides a pre-submission validation pass.
- `tests/` covers models, environment behavior, graders, and API endpoints.

The current default domain is a classification problem with `easy`, `medium`, and `hard` tasks. On problem day, the main swap surface remains:

- `environment/tasks.py`
- `environment/graders.py`
- `environment/reward.py`
- `environment/models.py` if the observation or action schema needs to change
