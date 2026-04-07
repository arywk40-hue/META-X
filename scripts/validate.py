"""Pre-submission validation for the data-cleaning environment."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient
import yaml

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
    manifest = yaml.safe_load(manifest_text)
    _check("name: data-cleaning-env" in manifest_text, "manifest name is correct")
    _check("tags:" in manifest_text and "openenv" in manifest_text, "manifest includes openenv tag")
    manifest_task_ids = [task["id"] for task in manifest.get("tasks", [])]
    _check(len(manifest_task_ids) == len(TASKS), "manifest task count matches runtime task count")
    _check(set(manifest_task_ids) == set(TASKS), "manifest task ids match runtime task ids")


def validate_inference_script() -> None:
    inference_path = ROOT / "inference.py"
    _check(inference_path.exists(), "root inference.py exists")
    contents = inference_path.read_text()
    _check("API_BASE_URL" in contents, "inference.py references API_BASE_URL")
    _check("MODEL_NAME" in contents, "inference.py references MODEL_NAME")
    _check("HF_TOKEN" in contents, "inference.py references HF_TOKEN")
    # The hackathon runner contract is provider-agnostic; this repo's inference.py may choose
    # to require HF_TOKEN only. Don't enforce an OPENAI_API_KEY fallback string.
    _check("OpenAI(" in contents, "inference.py uses the OpenAI client")
    _check("[START]" in contents, "inference.py emits [START] logs")
    _check("[STEP]" in contents, "inference.py emits [STEP] logs")
    _check("[END]" in contents, "inference.py emits [END] logs")


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
        grade_episode("duplicate_outlier", []),
        grade_episode(
            "duplicate_outlier",
            [{"action": {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "duplicate"}}],
        ),
        grade_episode(
            "duplicate_outlier",
            [
                {"action": {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "duplicate"}},
                {"action": {"action_type": "flag_anomaly", "row_index": 2, "column": "amount", "new_value": None, "reason": "outlier"}},
            ],
        ),
        grade_episode(
            "duplicate_outlier",
            [
                {"action": {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "duplicate"}},
                {"action": {"action_type": "flag_anomaly", "row_index": 2, "column": "amount", "new_value": None, "reason": "outlier"}},
                {"action": {"action_type": "fix_value", "row_index": 3, "column": "amount", "new_value": 50.0, "reason": "negative amount"}},
                {"action": {"action_type": "standardize_format", "row_index": 5, "column": "status", "new_value": "completed", "reason": "normalize case"}},
            ],
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
    _check(len(tasks.json()) >= 5, "GET /tasks exposes the data-cleaning benchmark")
    _check(all("answer" not in task for task in tasks.json()), "GET /tasks does not leak answers")

    reset = client.post("/reset")
    _check(reset.status_code == 200, "POST /reset succeeds with no body")
    session_id = reset.json()["session_id"]

    step = client.post(
        f"/step?session_id={session_id}",
        json={
            "action_type": "flag_anomaly",
            "row_index": 0,
            "column": "reading",
            "new_value": None,
            "reason": "This looks suspicious.",
        },
    )
    _check(step.status_code == 200, "POST /step succeeds with raw data-cleaning action JSON")

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

        websocket.send_json({"type": "reset", "payload": {"task_id": "null_filling"}})
        reset_message = websocket.receive_json()
        _check(reset_message["type"] == "reset", "WS /ws reset succeeds")

        websocket.send_json(
            {
                "type": "step",
                "payload": {
                    "action_type": "fill_missing",
                    "row_index": 1,
                    "column": "age",
                    "new_value": "unknown",
                    "reason": "age is missing",
                },
            }
        )
        step_message = websocket.receive_json()
        _check(step_message["type"] == "step", "WS /ws step succeeds")


def validate_baseline() -> None:
    agent = BaselineAgent(env=OpenEnv(), use_llm=False)
    results = agent.run(task_ids=["null_filling"], max_episodes_per_task=1)
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
