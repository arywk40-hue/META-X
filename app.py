"""FastAPI entrypoint exposing the OpenEnv HTTP interface."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from baseline.inference import run_baseline
from environment import TASKS, Action, OpenEnv, get_task, grade_episode
from environment.models import BaselineRequest, GraderRequest, ResetRequest


APP_VERSION = "1.0.0"
DEFAULT_MAX_CONCURRENT_ENVS = 100


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_to_grade(score: float) -> str:
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"


def _grader_breakdown(task_id: str, episode_history: list[dict[str, Any]], score: float) -> dict[str, float]:
    if task_id == "easy_01":
        total = sum(1 for step in episode_history if not step.get("evaluation", {}).get("skipped"))
        return {
            "correctness": score,
            "efficiency": min(1.0, total / max(1, len(episode_history))),
            "completeness": min(1.0, len(episode_history) / 10),
        }
    if task_id == "medium_01":
        violations = sum(1 for step in episode_history if step.get("evaluation", {}).get("constraint_violation"))
        return {
            "correctness": score,
            "efficiency": max(0.0, 1.0 - (violations * 0.1)),
            "completeness": min(1.0, len(episode_history) / 8),
        }
    verify_used = sum(1 for step in episode_history if step.get("action", {}).get("action_type") == "verify")
    return {
        "correctness": score,
        "efficiency": max(0.0, 1.0 - (verify_used * 0.1)),
        "completeness": min(1.0, len(episode_history) / 10),
    }


def _feedback_for_score(score: float, task_name: str) -> str:
    if score >= 0.85:
        return f"Strong performance on {task_name}. The agent handled the task reliably."
    if score >= 0.6:
        return f"Solid partial credit on {task_name}. There is room to tighten accuracy and efficiency."
    if score >= 0.3:
        return f"The run made some progress on {task_name}, but the grader saw notable mistakes or inefficiency."
    return f"The run struggled on {task_name}. Review the action formatting and the task constraints."


def _runtime_config() -> dict[str, Any]:
    return {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "7860")),
        "workers": int(os.getenv("WORKERS", "1")),
        "max_concurrent_envs": int(os.getenv("MAX_CONCURRENT_ENVS", str(DEFAULT_MAX_CONCURRENT_ENVS))),
    }


def create_app(
    environment: OpenEnv | None = None,
    max_concurrent_envs: int | None = None,
) -> FastAPI:
    """App factory used by the runtime and tests."""
    environment_type = (environment or OpenEnv()).__class__
    runtime_config = _runtime_config()
    app = FastAPI(
        title="OpenEnv RL Environment Template",
        version=APP_VERSION,
        description="Meta PyTorch Hackathon OpenEnv-compatible environment scaffold.",
    )
    app.state.sessions: dict[str, OpenEnv] = {}
    app.state.websocket_sessions: set[str] = set()
    app.state.runtime_config = {
        **runtime_config,
        "max_concurrent_envs": max_concurrent_envs or runtime_config["max_concurrent_envs"],
    }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_session_env(session_id: str | None) -> OpenEnv:
        if not session_id or session_id not in app.state.sessions:
            raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
        return app.state.sessions[session_id]

    def task_summaries() -> list[dict[str, Any]]:
        action_schema = Action.model_json_schema()
        return [{**task.summary(), "action_schema": action_schema} for task in TASKS.values()]

    def grade_payload(task_id: str, episode: list[dict[str, Any]]) -> dict[str, Any]:
        if not episode:
            raise HTTPException(status_code=400, detail="Episode history is empty or malformed")
        task = get_task(task_id)
        score = grade_episode(task_id, episode)
        breakdown = _grader_breakdown(task_id, episode, score)
        return {
            "task_id": task_id,
            "score": score,
            "max_possible_score": 1.0,
            "grade": _score_to_grade(score),
            "breakdown": breakdown,
            "feedback": _feedback_for_score(score, task.name),
            "percentile_estimate": int(round(score * 100)),
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "environment": "openenv-template",
            "version": APP_VERSION,
            "timestamp": _utc_timestamp(),
            "tasks_loaded": len(TASKS),
            "ws_endpoint": "/ws",
            "max_concurrent_envs": app.state.runtime_config["max_concurrent_envs"],
            "ready": True,
        }

    @app.post("/reset")
    def reset(request: ResetRequest | None = None, session_id: str | None = None) -> dict[str, Any]:
        try:
            request = request or ResetRequest()
            active_session_id = session_id or str(uuid4())
            env = environment_type()
            app.state.sessions[active_session_id] = env
            observation = env.reset(task_id=request.task_id, seed=request.seed)
            task = get_task(observation.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "session_id": active_session_id,
            "observation": observation.model_dump(mode="json"),
            "task_info": task.summary(),
        }

    @app.post("/step")
    def step(action: Action, session_id: str | None = None) -> dict[str, Any]:
        env = get_session_env(session_id)
        try:
            observation, reward, done, info = env.step(action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {
            "observation": observation.model_dump(mode="json"),
            "reward": reward.model_dump(mode="json"),
            "done": done,
            "info": info,
        }

    @app.get("/state")
    def state(session_id: str | None = None) -> dict[str, Any]:
        env = get_session_env(session_id)
        try:
            return env.state()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/tasks")
    def tasks() -> list[dict[str, Any]]:
        return task_summaries()

    @app.post("/grader")
    def grader(request: GraderRequest, session_id: str | None = None) -> dict[str, Any]:
        get_session_env(session_id)
        try:
            return grade_payload(request.task_id, request.episode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/baseline")
    def baseline(request: BaselineRequest) -> dict[str, Any]:
        try:
            return run_baseline(
                task_ids=request.task_ids,
                model=request.model,
                max_episodes_per_task=request.max_episodes_per_task,
                verbose=request.verbose,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive surface for remote APIs
            raise HTTPException(status_code=500, detail=f"Baseline inference failed: {exc}") from exc

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if len(app.state.websocket_sessions) >= app.state.runtime_config["max_concurrent_envs"]:
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Max concurrent envs reached")
            return

        connection_id = str(uuid4())
        env = environment_type()
        app.state.websocket_sessions.add(connection_id)
        await websocket.accept()

        try:
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type")
                payload = data.get("payload", {}) or {}

                if message_type == "reset":
                    request = ResetRequest.model_validate(payload)
                    observation = env.reset(task_id=request.task_id, seed=request.seed)
                    task = get_task(observation.task_id)
                    await websocket.send_json(
                        {
                            "type": "reset",
                            "observation": observation.model_dump(mode="json"),
                            "task_info": task.summary(),
                        }
                    )
                    continue

                if message_type == "step":
                    action = Action.model_validate(payload)
                    observation, reward, done, info = env.step(action)
                    await websocket.send_json(
                        {
                            "type": "step",
                            "observation": observation.model_dump(mode="json"),
                            "reward": reward.model_dump(mode="json"),
                            "done": done,
                            "info": info,
                        }
                    )
                    continue

                if message_type == "state":
                    await websocket.send_json({"type": "state", "state": env.state()})
                    continue

                if message_type == "tasks":
                    await websocket.send_json({"type": "tasks", "tasks": task_summaries()})
                    continue

                if message_type == "grader":
                    request = GraderRequest.model_validate(payload)
                    graded = grade_payload(request.task_id, request.episode)
                    await websocket.send_json({"type": "grader", **graded})
                    continue

                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": "Unsupported websocket message type. Use reset, step, state, tasks, or grader.",
                    }
                )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        finally:
            app.state.websocket_sessions.discard(connection_id)

    return app


app = create_app()
