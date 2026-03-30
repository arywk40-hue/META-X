"""Pre-submission validation for the code review environment."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app
from baseline.inference import BaselineAgent
from environment import OpenEnv, TASKS, grade_episode


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def validate_manifest() -> None:
    manifest_path = ROOT / "openenv.yaml"
    _check(manifest_path.exists(), "openenv.yaml exists")
    manifest_text = manifest_path.read_text()
    _check("name: openenv-template" in manifest_text, "manifest name is correct")
    _check("tags:" in manifest_text and "openenv" in manifest_text, "manifest includes openenv tag")
    _check("tasks:" in manifest_text and manifest_text.count("- id:") >= 3, "manifest lists tasks")


def validate_inference_script() -> None:
    inference_path = ROOT / "inference.py"
    _check(inference_path.exists(), "root inference.py exists")
    contents = inference_path.read_text()
    _check("API_BASE_URL" in contents, "inference.py references API_BASE_URL")
    _check("MODEL_NAME" in contents, "inference.py references MODEL_NAME")
    _check("HF_TOKEN" in contents, "inference.py references HF_TOKEN")
    _check("OPENAI_API_KEY" in contents, "inference.py supports OPENAI_API_KEY fallback")
    _check("OpenAI(" in contents, "inference.py uses the OpenAI client")


def validate_dockerfile() -> None:
    dockerfile = ROOT / "Dockerfile"
    _check(dockerfile.exists(), "Dockerfile exists")
    contents = dockerfile.read_text()
    _check("7860" in contents, "Dockerfile exposes HuggingFace Spaces port 7860")
    _check("uvicorn" in contents, "Dockerfile launches uvicorn")


def validate_graders() -> None:
    for task_id in TASKS:
        score = grade_episode(task_id, [])
        _check(0.0 <= score <= 1.0, f"{task_id} grader returns a valid score on empty history")

    score_levels = {
        grade_episode("security_vulnerability", []),
        grade_episode(
            "security_vulnerability",
            [{"action": {"bug_line": 1, "bug_type": "syntax", "explanation": "wrong"}}],
        ),
        grade_episode(
            "security_vulnerability",
            [{"action": {"bug_line": 23, "bug_type": "logic", "explanation": "wrong"}}],
        ),
        grade_episode(
            "security_vulnerability",
            [{"action": {"bug_line": 23, "bug_type": "security", "explanation": "This endpoint is unsafe and should be reviewed carefully."}}],
        ),
        grade_episode(
            "security_vulnerability",
            [{"action": {"bug_line": 23, "bug_type": "security", "explanation": "This is SQL injection from an f-string query instead of a parameterized query."}}],
        ),
    }
    _check(len(score_levels) >= 4, "grader exposes multiple distinct score levels")


def validate_api() -> None:
    client = TestClient(create_app(OpenEnv()))

    health = client.get("/health")
    _check(health.status_code == 200, "GET /health succeeds")

    tasks = client.get("/tasks")
    _check(tasks.status_code == 200, "GET /tasks succeeds")
    _check(isinstance(tasks.json(), list), "GET /tasks returns a bare array")
    _check(all("answer" not in task for task in tasks.json()), "GET /tasks does not leak answers")

    reset = client.post("/reset")
    _check(reset.status_code == 200, "POST /reset succeeds with no body")
    session_id = reset.json()["session_id"]

    step = client.post(
        f"/step?session_id={session_id}",
        json={
            "bug_line": 1,
            "bug_type": "syntax",
            "explanation": "This is probably wrong.",
        },
    )
    _check(step.status_code == 200, "POST /step succeeds with raw CodeReviewAction JSON")

    state = client.get(f"/state?session_id={session_id}")
    _check(state.status_code == 200, "GET /state succeeds")
    history = state.json()["episode_history"]
    _check(len(history) == 1, "state includes episode history")

    grader = client.post(
        f"/grader?session_id={session_id}",
        json={"task_id": state.json()["task_id"], "episode": history},
    )
    _check(grader.status_code == 200, "POST /grader succeeds")

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "tasks"})
        tasks_message = websocket.receive_json()
        _check(tasks_message["type"] == "tasks", "WS /ws tasks message succeeds")

        websocket.send_json({"type": "reset", "payload": {"task_id": "runtime_bug"}})
        reset_message = websocket.receive_json()
        _check(reset_message["type"] == "reset", "WS /ws reset succeeds")

        websocket.send_json(
            {
                "type": "step",
                "payload": {
                    "bug_line": 6,
                    "bug_type": "runtime",
                    "explanation": "The code divides by zero on an empty list.",
                },
            }
        )
        step_message = websocket.receive_json()
        _check(step_message["type"] == "step", "WS /ws step succeeds")


def validate_baseline() -> None:
    agent = BaselineAgent(env=OpenEnv(), use_llm=False)
    results = agent.run(task_ids=["runtime_bug"], max_episodes_per_task=1)
    _check(results["summary"]["total_episodes"] == 1, "baseline smoke test completes")


def main() -> None:
    validate_manifest()
    validate_inference_script()
    validate_dockerfile()
    validate_graders()
    validate_api()
    validate_baseline()
    print("Validation complete.")


if __name__ == "__main__":
    main()
