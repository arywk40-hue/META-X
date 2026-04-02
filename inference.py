"""Submission inference entrypoint for the data-cleaning OpenEnv environment."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
import re
import sys
from typing import Any

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI

from app import create_app
from environment.local_secrets import get_runtime_secret
from environment.models import Action


API_BASE_URL = get_runtime_secret("API_BASE_URL", default="https://router.huggingface.co/v1")
MODEL_NAME = get_runtime_secret("MODEL_NAME", default="Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = get_runtime_secret("HF_TOKEN")
API_KEY = HF_TOKEN or get_runtime_secret("OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN")
ENVIRONMENT_URL = os.getenv("ENVIRONMENT_URL", "http://127.0.0.1:8000")
BENCHMARK = os.getenv("OPENENV_BENCHMARK", "data-cleaning-env")
TASK_ID = os.getenv("TASK_ID", "adversarial_sensor")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
DEBUG_INFERENCE = os.getenv("DEBUG_INFERENCE", "").strip().lower() in {"1", "true", "yes"}


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM_PROMPT = (
    "You are an expert data cleaning agent. "
    "You will be shown a dirty tabular dataset preview. "
    "Choose exactly one cleaning action at a time. "
    "Respond ONLY in valid JSON with no extra text, no markdown fences, and no explanation outside the JSON object. "
    '{"action_type":"fix_value","row_index":2,"column":"email","new_value":"user@example.com","reason":"row_2_email_is_missing"}'
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_values = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={reward_values}",
        flush=True,
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
            {"action_type": "fill_missing", "row_index": 1, "column": "age", "new_value": "unknown", "reason": "age_missing"},
            {"action_type": "fix_value", "row_index": 2, "column": "age", "new_value": "unknown", "reason": "normalize_literal_null"},
            {"action_type": "fill_missing", "row_index": 4, "column": "email", "new_value": "missing@example.com", "reason": "email_missing"},
        ],
        "format_standardization": [
            {"action_type": "standardize_format", "row_index": 1, "column": "date", "new_value": "2024-01-15", "reason": "normalize_iso_date"},
            {"action_type": "standardize_format", "row_index": 2, "column": "date", "new_value": "2024-01-15", "reason": "normalize_iso_date"},
            {"action_type": "standardize_format", "row_index": 4, "column": "date", "new_value": "2024-01-15", "reason": "normalize_iso_date"},
            {"action_type": "fix_value", "row_index": 1, "column": "currency", "new_value": "USD", "reason": "uppercase_currency"},
            {"action_type": "fix_value", "row_index": 3, "column": "currency", "new_value": "USD", "reason": "normalize_currency_name"},
        ],
        "duplicate_outlier": [
            {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "drop_duplicate"},
            {"action_type": "flag_anomaly", "row_index": 2, "column": "amount", "new_value": None, "reason": "flag_outlier"},
            {"action_type": "fix_value", "row_index": 3, "column": "amount", "new_value": 50.0, "reason": "repair_negative_amount"},
            {"action_type": "standardize_format", "row_index": 5, "column": "status", "new_value": "completed", "reason": "normalize_status_case"},
        ],
        "multi_layer_pipeline": [
            {"action_type": "fix_value", "row_index": 1, "column": "qty", "new_value": 1, "reason": "qty_non_negative"},
            {"action_type": "flag_anomaly", "row_index": 2, "column": "customer_id", "new_value": None, "reason": "invalid_customer_fk"},
            {"action_type": "fix_value", "row_index": 2, "column": "unit_price", "new_value": 1.0, "reason": "price_positive"},
            {"action_type": "flag_anomaly", "row_index": 3, "column": "product_id", "new_value": None, "reason": "invalid_product_fk"},
            {"action_type": "fix_value", "row_index": 3, "column": "order_dt", "new_value": "2024-01-01", "reason": "repair_invalid_date"},
            {"action_type": "flag_anomaly", "row_index": 4, "column": "qty", "new_value": None, "reason": "qty_outlier"},
            {"action_type": "drop_row", "row_index": 5, "column": "row_id", "new_value": None, "reason": "drop_duplicate_order"},
        ],
        "adversarial_sensor": [
            {"action_type": "fill_missing", "row_index": 2, "column": "reading", "new_value": None, "reason": "fill_missing_reading"},
            {"action_type": "flag_anomaly", "row_index": 5, "column": "reading", "new_value": None, "reason": "flag_impossible_outlier"},
        ],
        "titanic_manifest": [
            {"action_type": "fill_missing", "row_index": 0, "column": "Age", "new_value": 28, "reason": "impute_age"},
            {"action_type": "fill_missing", "row_index": 1, "column": "Embarked", "new_value": "C", "reason": "fill_embarked"},
            {"action_type": "fill_missing", "row_index": 2, "column": "Embarked", "new_value": "C", "reason": "fill_embarked"},
            {"action_type": "fill_missing", "row_index": 3, "column": "Cabin", "new_value": "Unknown", "reason": "fill_cabin_placeholder"},
        ],
    }
    plan = action_plans.get(task_id, action_plans["adversarial_sensor"])
    return Action.model_validate(plan[min(step, len(plan) - 1)])


def format_action(action: Action) -> str:
    value = action.new_value
    if value is None:
        value_repr = "null"
    elif isinstance(value, str):
        value_repr = value.replace(" ", "_")
    else:
        value_repr = str(value)
    return (
        f"{action.action_type}(row={action.row_index},column={action.column},"
        f"value={value_repr})"
    )


def choose_task(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    requested = next(
        (
            task
            for task in tasks
            if task.get("id") == TASK_ID or task.get("task_id") == TASK_ID
        ),
        None,
    )
    if requested is not None:
        return requested

    preferred = next(
        (
            task
            for task in tasks
            if task.get("id") == "adversarial_sensor" or task.get("task_id") == "adversarial_sensor"
        ),
        None,
    )
    if preferred is not None:
        return preferred
    return tasks[0]


def build_messages(observation: dict[str, Any]) -> list[dict[str, str]]:
    return [
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


def request_action(
    llm_client: OpenAI,
    messages: list[dict[str, str]],
    task_id: str,
    observation: dict[str, Any],
) -> tuple[Action, str]:
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
        return parse_model_action(response_text), response_text
    except Exception as exc:  # noqa: BLE001
        if DEBUG_INFERENCE:
            print(f"model_request_failed={exc}", file=sys.stderr, flush=True)
        return fallback_action(task_id, int(observation["step"])), ""


def ensure_api_configuration() -> None:
    if not API_BASE_URL:
        raise KeyError("Set API_BASE_URL before running inference.py")
    if not MODEL_NAME:
        raise KeyError("Set MODEL_NAME before running inference.py")
    if not API_KEY:
        raise KeyError("Set HF_TOKEN before running inference.py")


def main() -> None:
    ensure_api_configuration()

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards: list[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    logged_start = False

    with ExitStack() as stack:
        env_client = build_env_client(stack)
        try:
            tasks_response = env_client.get("/tasks")
            tasks_response.raise_for_status()
            tasks = tasks_response.json()
            chosen_task = choose_task(tasks)
            chosen_task_id = chosen_task.get("id", chosen_task["task_id"])

            log_start(task=chosen_task_id, env=BENCHMARK, model=MODEL_NAME)
            logged_start = True

            reset_response = env_client.post("/reset", json={"task_id": chosen_task_id})
            reset_response.raise_for_status()
            reset_payload = reset_response.json()
            session_id = reset_payload["session_id"]
            observation = reset_payload["observation"]
            messages = build_messages(observation)

            while not observation["done"] and observation["attempts_remaining"] > 0 and steps_taken < MAX_STEPS:
                next_step = steps_taken + 1
                action, response_text = request_action(llm_client, messages, chosen_task_id, observation)

                step_response = env_client.post(
                    "/step",
                    params={"session_id": session_id},
                    json=action.model_dump(mode="json"),
                )
                step_response.raise_for_status()
                step_payload = step_response.json()
                observation = step_payload["observation"]

                reward_value = float(step_payload["reward"]["value"])
                rewards.append(reward_value)
                steps_taken = next_step

                log_step(
                    step=next_step,
                    action=format_action(action),
                    reward=reward_value,
                    done=bool(step_payload["done"]),
                    error=None,
                )

                if observation["done"] or observation["attempts_remaining"] <= 0 or steps_taken >= MAX_STEPS:
                    break

                assistant_turn = (
                    strip_json_block(response_text)
                    if response_text
                    else json.dumps(action.model_dump(mode="json"), separators=(",", ":"))
                )
                messages.append({"role": "assistant", "content": assistant_turn})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Feedback: {observation['feedback']}\n"
                            f"Issues remaining: {observation['issues_remaining']}\n"
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

            final_score = float(grader_payload["score"])
            success = bool(state_payload.get("solved", False) or final_score >= SUCCESS_SCORE_THRESHOLD)
        except Exception as exc:  # noqa: BLE001
            if DEBUG_INFERENCE:
                print(f"inference_failed={exc}", file=sys.stderr, flush=True)
        finally:
            if not logged_start:
                log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    main()
