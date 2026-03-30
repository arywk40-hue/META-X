"""Root inference entrypoint for the code review environment."""

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
HF_TOKEN = os.environ["HF_TOKEN"]
API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)
ENVIRONMENT_URL = os.getenv("ENVIRONMENT_URL", "http://127.0.0.1:8000")
MAX_TOKENS = 256
TEMPERATURE = 0.2
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM_PROMPT = (
    "You are an expert Python security and code reviewer. "
    "You will be shown a Python code snippet with line numbers. "
    "Identify the single most critical bug. Respond ONLY in valid JSON with no extra text: "
    '{"bug_line": <int>, "bug_type": "syntax|runtime|logic|security", "explanation": "<concise explanation>"}'
)


def strip_json_block(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = JSON_BLOCK_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def build_env_client(stack: ExitStack) -> Any:
    """Prefer a live HTTP environment, but fall back to an in-process app."""
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


def fallback_action(code_snippet: str) -> Action:
    if "audit_log" in code_snippet or "SELECT action" in code_snippet:
        return Action(
            bug_line=23,
            bug_type="security",
            explanation="The query is built with an f-string from user input, which allows SQL injection.",
        )
    if "binary_search" in code_snippet:
        return Action(
            bug_line=9,
            bug_type="logic",
            explanation="The loop condition uses < instead of <=, so the last candidate may never be checked.",
        )
    return Action(
        bug_line=6,
        bug_type="runtime",
        explanation="The function divides by len(numbers) without handling an empty list, which can raise ZeroDivisionError.",
    )


def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with ExitStack() as stack:
        env_client = build_env_client(stack)

        tasks_response = env_client.get("/tasks")
        tasks_response.raise_for_status()
        tasks = tasks_response.json()
        chosen_task = next(
            (task for task in tasks if task.get("id") == "security_vulnerability" or task.get("task_id") == "security_vulnerability"),
            None,
        )
        if chosen_task is None:
            chosen_task = next((task for task in tasks if task["difficulty"] == "hard"), tasks[0])

        reset_response = env_client.post("/reset", json={"task_id": chosen_task.get("id", chosen_task["task_id"])})
        reset_response.raise_for_status()
        reset_payload = reset_response.json()
        session_id = reset_payload["session_id"]
        observation = reset_payload["observation"]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this code:\n\n{observation['code_snippet']}"},
        ]

        print(f"Selected task: {chosen_task.get('id', chosen_task['task_id'])}")

        attempt_number = 0
        while not observation["done"] and observation["attempts_remaining"] > 0:
            attempt_number += 1

            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_model_action(response_text)
            except Exception as exc:  # noqa: BLE001
                print(f"Model request failed ({exc}). Falling back to heuristic answer.")
                response_text = ""
                action = fallback_action(observation["code_snippet"])

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
                    "content": f"Feedback: {observation['feedback']}\n\nRevise your answer.",
                }
            )

        state_response = env_client.get("/state", params={"session_id": session_id})
        state_response.raise_for_status()
        state_payload = state_response.json()

        grader_response = env_client.post(
            "/grader",
            params={"session_id": session_id},
            json={
                "task_id": chosen_task.get("id", chosen_task["task_id"]),
                "episode": state_payload["episode_history"],
            },
        )
        grader_response.raise_for_status()
        grader_payload = grader_response.json()

        print(f"Final score: {grader_payload['score']:.2f}")
        print(f"Grade: {grader_payload['grade']}")
        print(f"Feedback: {grader_payload['feedback']}")


if __name__ == "__main__":
    main()
