"""Integration tests for the FastAPI API surface."""

import pytest

from environment.tasks import TASKS


def test_health_endpoint(test_client) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["ready"] is True


def test_tasks_endpoint_lists_all_tasks_without_answers(test_client) -> None:
    response = test_client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == len(TASKS)
    assert len(payload) >= 5
    assert payload[0]["action_schema"]["required"] == ["action_type", "column", "reason"]
    assert "answer" not in payload[0]


def test_reset_endpoint_returns_observation_and_supports_empty_body(test_client) -> None:
    response = test_client.post("/reset")
    assert response.status_code == 200
    payload = response.json()
    assert "session_id" in payload
    assert payload["observation"]["task_id"] in TASKS
    assert "dataset_preview" in payload["observation"]
    assert payload["observation"]["done"] is False


def test_reset_endpoint_rejects_invalid_task(test_client) -> None:
    response = test_client.post("/reset", json={"task_id": "missing_task"})
    assert response.status_code == 400


def test_step_and_state_endpoints_work_together(test_client) -> None:
    reset = test_client.post("/reset", json={"task_id": "null_filling"})
    session_id = reset.json()["session_id"]
    step = test_client.post(
        f"/step?session_id={session_id}",
        json={
            "action_type": "fill_missing",
            "row_index": 1,
            "column": "age",
            "new_value": "unknown",
            "reason": "Row 1 age is missing.",
        },
    )
    state = test_client.get(f"/state?session_id={session_id}")

    assert step.status_code == 200
    assert step.json()["reward"]["value"] > 0.0
    assert step.json()["observation"]["done"] is False
    assert step.json()["observation"]["issues_remaining"] == 2
    assert state.status_code == 200
    assert state.json()["step_count"] == 1


def test_state_endpoint_requires_active_episode(test_client) -> None:
    response = test_client.get("/state")
    assert response.status_code == 404


def test_grader_endpoint_scores_episode_history(test_client) -> None:
    reset = test_client.post("/reset", json={"task_id": "adversarial_sensor"})
    session_id = reset.json()["session_id"]
    test_client.post(
        f"/step?session_id={session_id}",
        json={
            "action_type": "fill_missing",
            "row_index": 2,
            "column": "reading",
            "new_value": None,
            "reason": "The reading is missing.",
        },
    )
    history = test_client.get(f"/state?session_id={session_id}").json()["episode_history"]

    response = test_client.post(
        f"/grader?session_id={session_id}",
        json={"task_id": "adversarial_sensor", "episode": history},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["score"] <= 1.0
    assert "breakdown" in payload


def test_baseline_endpoint_runs_smoke_test(test_client) -> None:
    response = test_client.post(
        "/baseline",
        json={"task_ids": ["null_filling"], "max_episodes_per_task": 1, "verbose": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_episodes"] == 1
    assert payload["results"][0]["task_id"] == "null_filling"


def test_websocket_supports_reset_step_and_state(test_client) -> None:
    with test_client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "payload": {"task_id": "null_filling"}})
        reset_message = websocket.receive_json()
        assert reset_message["type"] == "reset"
        assert reset_message["observation"]["task_id"] == "null_filling"

        websocket.send_json(
            {
                "type": "step",
                "payload": {
                    "action_type": "fill_missing",
                    "row_index": 1,
                    "column": "age",
                    "new_value": "unknown",
                    "reason": "The age value is missing.",
                },
            }
        )
        step_message = websocket.receive_json()
        assert step_message["type"] == "step"
        assert "reward" in step_message

        websocket.send_json({"type": "state"})
        state_message = websocket.receive_json()
        assert state_message["type"] == "state"
        assert state_message["state"]["step_count"] == 1


def test_websocket_honors_max_concurrent_envs() -> None:
    from fastapi.testclient import TestClient

    from app import create_app

    with TestClient(create_app(max_concurrent_envs=1)) as limited_client:
        with limited_client.websocket_connect("/ws") as first_ws:
            first_ws.send_json({"type": "tasks"})
            assert first_ws.receive_json()["type"] == "tasks"

            with pytest.raises(Exception):
                with limited_client.websocket_connect("/ws"):
                    pass
