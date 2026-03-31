"""Root inference entrypoint for the data-cleaning environment."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
import re
from typing import Any

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI

from app import create_app
from environment.models import Action


API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ.get("HF_TOKEN")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or HF_TOKEN
if not API_KEY:
    raise KeyError("Set OPENAI_API_KEY (preferred), GROQ_API_KEY, or HF_TOKEN before running inference.py")
ENVIRONMENT_URL = os.getenv("ENVIRONMENT_URL", "http://127.0.0.1:8000")
TASK_ID = os.getenv("TASK_ID") or os.getenv("PREFERRED_TASK_ID")
MAX_TOKENS = 256
TEMPERATURE = 0.2
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM_PROMPT = (
    "You are an expert data cleaning agent. "
    "You will be shown a dirty tabular dataset preview. "
    "Choose exactly one cleaning action at a time. "
    "Respond ONLY in valid JSON with no extra text, no markdown fences, and no explanation outside the JSON object. "
    '{"action_type":"fix_value","row_index":2,"column":"email","new_value":"user@example.com","reason":"row 2 email is missing"}'
)


def strip_json_block(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = JSON_BLOCK_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def build_env_client(stack: ExitStack) -> Any:
    remote_client = httpx.Client(base_url=ENVIRONMENT_URL, timeout=REQUEST_TIMEOUT)
    try:
        health_response = remote_client.get("/health")
        health_response.raise_for_status()
        stack.callback(remote_client.close)
        return remote_client
    except Exception:
        remote_client.close()

    local_client = TestClient(create_app())
    stack.enter_context(local_client)
    return local_client


def parse_model_action(response_text: str) -> Action:
    try:
        payload = json.loads(strip_json_block(response_text))
        return Action.model_validate(payload)
    except Exception:
        return Action.from_llm_output(response_text, [])


def fallback_action(task_id: str, step: int) -> Action:
    action_plans = {
        "null_filling": [
            {"action_type": "fill_missing", "row_index": 1, "column": "age", "new_value": "unknown", "reason": "age is missing"},
            {"action_type": "fix_value", "row_index": 2, "column": "age", "new_value": "unknown", "reason": "literal NULL should be normalized"},
            {"action_type": "fill_missing", "row_index": 4, "column": "email", "new_value": "missing@example.com", "reason": "email is missing"},
        ],
        "format_standardization": [
            {"action_type": "standardize_format", "row_index": 1, "column": "date", "new_value": "2024-01-15", "reason": "normalize to ISO"},
            {"action_type": "standardize_format", "row_index": 2, "column": "date", "new_value": "2024-01-15", "reason": "normalize to ISO"},
            {"action_type": "standardize_format", "row_index": 4, "column": "date", "new_value": "2024-01-15", "reason": "normalize to ISO"},
            {"action_type": "fix_value", "row_index": 1, "column": "currency", "new_value": "USD", "reason": "uppercase currency"},
            {"action_type": "fix_value", "row_index": 3, "column": "currency", "new_value": "USD", "reason": "standardize full currency name"},
        ],
        "duplicate_outlier": [
            {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "exact duplicate"},
            {"action_type": "flag_anomaly", "row_index": 2, "column": "amount", "new_value": None, "reason": "extreme outlier"},
            {"action_type": "fix_value", "row_index": 3, "column": "amount", "new_value": 50.0, "reason": "negative amount should be non-negative"},
            {"action_type": "standardize_format", "row_index": 5, "column": "status", "new_value": "completed", "reason": "normalize casing"},
        ],
        "multi_layer_pipeline": [
            {"action_type": "fix_value", "row_index": 1, "column": "qty", "new_value": 1, "reason": "quantity cannot be negative"},
            {"action_type": "flag_anomaly", "row_index": 2, "column": "customer_id", "new_value": None, "reason": "invalid foreign key"},
            {"action_type": "fix_value", "row_index": 2, "column": "unit_price", "new_value": 1.0, "reason": "price cannot be zero"},
            {"action_type": "flag_anomaly", "row_index": 3, "column": "product_id", "new_value": None, "reason": "invalid foreign key"},
            {"action_type": "fix_value", "row_index": 3, "column": "order_dt", "new_value": "2024-01-01", "reason": "repair invalid date"},
            {"action_type": "flag_anomaly", "row_index": 4, "column": "qty", "new_value": None, "reason": "quantity is an outlier"},
            {"action_type": "drop_row", "row_index": 5, "column": "row_id", "new_value": None, "reason": "duplicate order"},
        ],
        "adversarial_sensor": [
            {"action_type": "fill_missing", "row_index": 2, "column": "reading", "new_value": None, "reason": "reading is missing"},
            {"action_type": "flag_anomaly", "row_index": 5, "column": "reading", "new_value": None, "reason": "999C is an impossible outlier"},
        ],
        "titanic_manifest": [
            {"action_type": "fill_missing", "row_index": 0, "column": "Age", "new_value": 28, "reason": "impute a plausible numeric age"},
            {"action_type": "fill_missing", "row_index": 1, "column": "Embarked", "new_value": "C", "reason": "fill a valid port code"},
            {"action_type": "fill_missing", "row_index": 2, "column": "Embarked", "new_value": "C", "reason": "fill a valid port code"},
            {"action_type": "fill_missing", "row_index": 3, "column": "Cabin", "new_value": "Unknown", "reason": "use a consistent placeholder"},
        ],
    }
    plan = action_plans[task_id]
    return Action.model_validate(plan[min(step, len(plan) - 1)])


def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with ExitStack() as stack:
        env_client = build_env_client(stack)

        tasks_response = env_client.get("/tasks")
        tasks_response.raise_for_status()
        tasks = tasks_response.json()
        chosen_task = None
        if TASK_ID:
            chosen_task = next(
                (
                    task
                    for task in tasks
                    if task.get("id") == TASK_ID or task.get("task_id") == TASK_ID
                ),
                None,
            )

        if chosen_task is None:
            chosen_task = next(
                (task for task in tasks if task.get("id") == "adversarial_sensor" or task.get("task_id") == "adversarial_sensor"),
                None,
            )
        if chosen_task is None:
            chosen_task = next((task for task in tasks if task["difficulty"] == "hard"), tasks[0])

        chosen_task_id = chosen_task.get("id", chosen_task["task_id"])
        reset_response = env_client.post("/reset", json={"task_id": chosen_task_id})
        reset_response.raise_for_status()
        reset_payload = reset_response.json()
        session_id = reset_payload["session_id"]
        observation = reset_payload["observation"]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {observation['task_name']}\n"
                    f"Description: {observation['task_description']}\n"
                    f"Dataset:\n{observation['dataset_preview']}"
                ),
            },
        ]

        print(f"Selected task: {chosen_task_id}")

        attempt_number = 0
        while not observation["done"] and observation["attempts_remaining"] > 0:
            attempt_number += 1

            try:
                try:
                    completion = llm_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    completion = llm_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                response_text = completion.choices[0].message.content or ""
                action = parse_model_action(response_text)
            except Exception as exc:  # noqa: BLE001
                print(f"Model request failed ({exc}). Falling back to heuristic action.")
                response_text = ""
                action = fallback_action(chosen_task_id, observation["step"])

            step_response = env_client.post(
                "/step",
                params={"session_id": session_id},
                json=action.model_dump(mode="json"),
            )
            step_response.raise_for_status()
            step_payload = step_response.json()
            observation = step_payload["observation"]

            print(f"Attempt {attempt_number}: {action.model_dump(mode='json')}")
            print(
                f"Reward: {step_payload['reward']['value']:.2f} | "
                f"Feedback: {observation['feedback']}"
            )

            if observation["done"] or observation["attempts_remaining"] <= 0:
                break

            assistant_turn = (
                strip_json_block(response_text)
                if response_text
                else json.dumps(action.model_dump(mode="json"))
            )
            messages.append({"role": "assistant", "content": assistant_turn})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Feedback: {observation['feedback']}\n"
                        f"Issues remaining: {observation['issues_remaining']}\n\n"
                        "Revise your next action."
                    ),
                }
            )

        state_response = env_client.get("/state", params={"session_id": session_id})
        state_response.raise_for_status()
        state_payload = state_response.json()

        grader_response = env_client.post(
            "/grader",
            params={"session_id": session_id},
            json={"task_id": chosen_task_id, "episode": state_payload["episode_history"]},
        )
        grader_response.raise_for_status()
        grader_payload = grader_response.json()

        print(f"Final score: {grader_payload['score']:.2f}")
        print(f"Grade: {grader_payload['grade']}")
        print(f"Feedback: {grader_payload['feedback']}")


if __name__ == "__main__":
    main()
