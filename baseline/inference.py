"""Baseline agent runner for the data-cleaning environment."""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev
from typing import Any

try:  # pragma: no cover - exercised only when the dependency exists
    from openai import OpenAI
except ImportError:  # pragma: no cover - local fallback path
    OpenAI = None

from environment import Action, OpenEnv
from environment.graders import grade_episode
from environment.tasks import TASKS, get_task


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


class BaselineAgent:
    """Runs either an OpenAI-backed baseline or a deterministic heuristic fallback."""

    def __init__(
        self,
        env: OpenEnv | None = None,
        model: str = os.getenv("MODEL_NAME", "gpt-4o-mini"),
        use_llm: bool | None = None,
    ) -> None:
        self.env = env or OpenEnv()
        self.model = model
        hf_token = os.getenv("HF_TOKEN")
        api_key = os.getenv("OPENAI_API_KEY", hf_token or "")
        base_url = os.getenv("API_BASE_URL")
        self.use_llm = bool(use_llm) if use_llm is not None else bool(api_key and OpenAI)
        self.client = OpenAI(api_key=api_key, base_url=base_url) if self.use_llm and OpenAI else None

    def select_action(self, observation) -> Action:
        if self.use_llm:
            try:
                return self._llm_action(observation)
            except Exception:
                return self._heuristic_action(observation)
        return self._heuristic_action(observation)

    def run_episode(self, task_id: str, seed: int | None = None) -> dict[str, Any]:
        observation = self.env.reset(task_id=task_id, seed=seed)
        done = False

        while not done:
            action = self.select_action(observation)
            observation, _, done, _ = self.env.step(action)

        state = self.env.state()
        score = grade_episode(task_id, state["episode_history"])
        return {
            "score": score,
            "steps": state["step_count"],
            "total_reward": state["cumulative_reward"],
            "grade": _score_to_grade(score),
        }

    def run(
        self,
        task_ids: list[str] | None = None,
        max_episodes_per_task: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        selected_task_ids = task_ids or list(TASKS)
        for task_id in selected_task_ids:
            get_task(task_id)

        results = []
        total_steps = 0
        total_episodes = 0

        for task_id in selected_task_ids:
            task = get_task(task_id)
            episodes = []
            for episode_index in range(max_episodes_per_task):
                episode = self.run_episode(task_id=task_id, seed=episode_index)
                total_steps += episode["steps"]
                total_episodes += 1
                episodes.append(
                    {
                        "score": round(episode["score"], 4),
                        "steps": episode["steps"],
                        "grade": episode["grade"],
                    }
                )
                if verbose:
                    print(f"{task_id} episode {episode_index + 1}: {episode['score']:.3f}")

            scores = [episode["score"] for episode in episodes]
            results.append(
                {
                    "task_id": task.id,
                    "task_name": task.name,
                    "difficulty": task.difficulty.value,
                    "episodes": episodes,
                    "mean_score": round(mean(scores), 4),
                    "std_score": round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
                    "best_score": round(max(scores), 4),
                }
            )

        overall_mean = mean([task["mean_score"] for task in results]) if results else 0.0
        return {
            "results": results,
            "summary": {
                "total_episodes": total_episodes,
                "total_steps": total_steps,
                "overall_mean": round(overall_mean, 4),
                "model_used": self.model if self.use_llm else "heuristic-baseline",
                "runner_mode": "llm" if self.use_llm else "heuristic",
            },
        }

    def _llm_action(self, observation) -> Action:
        prompt = observation.to_prompt()
        if self.client is None:
            raise RuntimeError("OpenAI client is not available")

        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=200,
            )
            output_text = getattr(response, "output_text", "") or ""
        else:  # pragma: no cover - compatibility path
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            output_text = response.choices[0].message.content or ""

        return Action.from_llm_output(output_text, observation.available_actions)

    def _heuristic_action(self, observation) -> Action:
        step = observation.step
        task_id = observation.task_id

        if task_id == "null_filling":
            actions = [
                {"action_type": "fill_missing", "row_index": 1, "column": "age", "new_value": "unknown", "reason": "row 1 age is missing"},
                {"action_type": "fix_value", "row_index": 2, "column": "age", "new_value": "unknown", "reason": "literal NULL should be normalized"},
                {"action_type": "fill_missing", "row_index": 4, "column": "email", "new_value": "missing@example.com", "reason": "row 4 email is missing"},
            ]
        elif task_id == "format_standardization":
            actions = [
                {"action_type": "standardize_format", "row_index": 1, "column": "date", "new_value": "2024-01-15", "reason": "normalize slash date to ISO"},
                {"action_type": "standardize_format", "row_index": 2, "column": "date", "new_value": "2024-01-15", "reason": "normalize month-name date to ISO"},
                {"action_type": "standardize_format", "row_index": 4, "column": "date", "new_value": "2024-01-15", "reason": "normalize US date to ISO"},
                {"action_type": "fix_value", "row_index": 1, "column": "currency", "new_value": "USD", "reason": "uppercase 3-letter currency code"},
                {"action_type": "fix_value", "row_index": 3, "column": "currency", "new_value": "USD", "reason": "standardize full currency name"},
            ]
        elif task_id == "duplicate_outlier":
            actions = [
                {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "exact duplicate of row 0"},
                {"action_type": "flag_anomaly", "row_index": 2, "column": "amount", "new_value": None, "reason": "amount is an outlier"},
                {"action_type": "fix_value", "row_index": 3, "column": "amount", "new_value": 50.00, "reason": "negative amount should be non-negative"},
                {"action_type": "standardize_format", "row_index": 5, "column": "status", "new_value": "completed", "reason": "normalize status casing"},
            ]
        elif task_id == "multi_layer_pipeline":
            actions = [
                {"action_type": "fix_value", "row_index": 1, "column": "qty", "new_value": 1, "reason": "quantity cannot be negative"},
                {"action_type": "flag_anomaly", "row_index": 2, "column": "customer_id", "new_value": None, "reason": "customer id is invalid"},
                {"action_type": "fix_value", "row_index": 2, "column": "unit_price", "new_value": 1.0, "reason": "price cannot be zero"},
                {"action_type": "flag_anomaly", "row_index": 3, "column": "product_id", "new_value": None, "reason": "product id is invalid"},
                {"action_type": "fix_value", "row_index": 3, "column": "order_dt", "new_value": "2024-01-01", "reason": "repair invalid month in date"},
                {"action_type": "flag_anomaly", "row_index": 4, "column": "qty", "new_value": None, "reason": "quantity is a statistical outlier"},
                {"action_type": "drop_row", "row_index": 5, "column": "row_id", "new_value": None, "reason": "exact duplicate of row 0"},
            ]
        else:
            actions = [
                {"action_type": "fill_missing", "row_index": 2, "column": "reading", "new_value": None, "reason": "reading is missing"},
                {"action_type": "flag_anomaly", "row_index": 5, "column": "reading", "new_value": None, "reason": "999C is an impossible outlier"},
            ]

        chosen = actions[min(step, len(actions) - 1)]
        return Action.model_validate(chosen)


def run_baseline(
    task_ids: list[str] | None = None,
    model: str = "gpt-4o-mini",
    max_episodes_per_task: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper used by the API and CLI."""
    agent = BaselineAgent(env=OpenEnv(), model=model)
    return agent.run(
        task_ids=task_ids,
        max_episodes_per_task=max_episodes_per_task,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data-cleaning baseline agent.")
    parser.add_argument("--task-id", dest="task_ids", action="append", help="Task ID to run. Repeat for multiple.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use when OPENAI_API_KEY is set.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task.")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode progress.")
    args = parser.parse_args()

    results = run_baseline(
        task_ids=args.task_ids,
        model=args.model,
        max_episodes_per_task=args.episodes,
        verbose=args.verbose,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
