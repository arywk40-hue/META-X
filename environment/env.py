"""Core OpenEnv implementation for the data-cleaning domain."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from datetime import datetime, timezone
import random
import re
import threading
from typing import Any, Callable
from uuid import uuid4

from .graders import grade_episode
from .models import Action, Observation, Reward, State, Task
from .reward import compute_reward
from .tasks import TASKS, get_task


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dataset_preview(dataset_preview: str) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [line.rstrip() for line in dataset_preview.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        return [], []

    headers = [segment.strip() for segment in lines[0].split("|")]
    rows: list[dict[str, Any]] = []
    for line in lines[2:]:
        values = [segment.strip() for segment in line.split("|")]
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        elif len(values) > len(headers):
            values = values[: len(headers)]
        row = {header: values[index] for index, header in enumerate(headers)}
        row["__row_status__"] = "active"
        rows.append(row)

    return headers, rows


def _render_dataset_preview(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not headers:
        return ""

    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered_row: list[str] = []
        dropped = row.get("__row_status__") == "dropped"
        for header in headers:
            if dropped and header != "row_id":
                rendered_row.append("<DROPPED>")
            else:
                rendered_row.append(str(row.get(header, "")))
        rendered_rows.append(rendered_row)

    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_line = " | ".join(f"{header:<{widths[index]}}" for index, header in enumerate(headers))
    separator_line = "-|-".join("-" * width for width in widths)
    row_lines = [
        " | ".join(f"{value:<{widths[index]}}" for index, value in enumerate(row))
        for row in rendered_rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


class OpenEnvBase(ABC):
    @abstractmethod
    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> dict[str, Any]:
        raise NotImplementedError


class OpenEnv(OpenEnvBase):
    """Stateful data-cleaning environment."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.custom_task: Task | None = None
        self.custom_grader: Callable[[list[dict[str, Any]]], float] | None = None
        self.current_task: Task | None = None
        self.episode_id: str | None = None
        self.episode_history: list[dict[str, Any]] = []
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.current_observation: Observation | None = None
        self.started_at: datetime | None = None
        self.last_action_at: datetime | None = None
        self.seed: int | None = None
        self.task_state: dict[str, Any] = {}

    def set_dynamic_task(
        self,
        task: Task,
        grader: Callable[[list[dict[str, Any]]], float],
    ) -> None:
        self.custom_task = task
        self.custom_grader = grader

    def resolve_task(self, task_id: str | None = None) -> Task:
        if self.custom_task is not None and (task_id is None or task_id == self.custom_task.id):
            return self.custom_task
        if task_id is None:
            raise ValueError("Task id is required when no dynamic task is attached.")
        return get_task(task_id)

    def grade_history(self, task_id: str, episode_history: list[dict[str, Any]]) -> float:
        if self.custom_task is not None and self.custom_task.id == task_id and self.custom_grader is not None:
            score = float(self.custom_grader(episode_history))
            return round(max(0.0, min(1.0, score)), 4)
        return grade_episode(task_id, episode_history)

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        with self._lock:
            task = self.resolve_task(task_id) if (task_id or self.custom_task is not None) else self._choose_task(seed)
            self.current_task = task
            self.episode_id = str(uuid4())
            self.seed = seed
            self.started_at = utc_now()
            self.last_action_at = None
            self.step_count = 0
            self.done = False
            self.cumulative_reward = 0.0
            self.episode_history = []

            issues = task.config.get("issues", [])
            traps = task.config.get("traps", [])
            self.task_state = {
                "attempts_remaining": task.max_steps,
                "feedback_history": [],
                "feedback": "",
                "solved": False,
                "final_score": 0.0,
                "fixed_issues": set(),
                "penalized_traps": set(),
                "total_issues": len(issues),
                "traps": set(traps),
            }
            headers, rows = _parse_dataset_preview(task.config.get("dataset_preview", ""))
            self.task_state["dataset_headers"] = headers
            self.task_state["dataset_rows"] = rows
            self.current_observation = self._build_observation()
            return self.current_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        with self._lock:
            if self.current_task is None:
                raise ValueError("No active episode. Call reset() first.")
            if self.done:
                raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

            state_before = self._reward_state()
            obs_before = self.current_observation or self._build_observation()
            transition = self._evaluate_action(action)

            self.step_count += 1
            self.last_action_at = utc_now()
            self.task_state["attempts_remaining"] = max(0, self.current_task.max_steps - self.step_count)
            self.task_state["feedback"] = transition["feedback"]
            self.task_state["feedback_history"].append(transition["feedback"])
            self.task_state["solved"] = transition["solved"]
            self.done = transition["solved"] or self.task_state["attempts_remaining"] == 0

            state_after = self._reward_state(transition=transition)
            reward = compute_reward(
                state_before,
                action,
                state_after,
                task_config={**self.current_task.config, "total_issues": self.task_state["total_issues"]},
            )
            self.cumulative_reward += reward.value

            info = self._build_info(transition)
            history_entry = {
                "step": self.step_count,
                "observation": obs_before.model_dump(mode="json"),
                "action": action.model_dump(mode="json"),
                "reward": reward.value,
                "reward_detail": reward.model_dump(mode="json"),
                "done": self.done,
                "info": info,
            }
            self.episode_history.append(history_entry)

            if self.done:
                final_score = self.grade_history(self.current_task.id, self.episode_history)
                self.task_state["final_score"] = final_score
                info["final_score"] = final_score

            self.current_observation = self._build_observation()
            return self.current_observation, reward, self.done, info

    def state(self) -> dict[str, Any]:
        with self._lock:
            if self.current_task is None or self.current_observation is None or self.episode_id is None:
                raise ValueError("No active episode. Call reset() to start.")

            state = State(
                episode_id=self.episode_id,
                step_count=self.step_count,
                task_id=self.current_task.id,
                solved=bool(self.task_state.get("solved", False)),
                final_score=float(self.task_state.get("final_score", 0.0)),
            )
            payload = state.model_dump(mode="json")
            payload.update(
                {
                    "task_name": self.current_task.name,
                    "difficulty": self.current_task.difficulty.value,
                    "done": self.done,
                    "attempts_remaining": self.task_state.get("attempts_remaining", 0),
                    "cumulative_reward": round(self.cumulative_reward, 4),
                    "issues_remaining": self._issues_remaining(),
                    "current_observation": self.current_observation.model_dump(mode="json"),
                    "episode_history": copy.deepcopy(self.episode_history),
                    "metadata": {
                        "started_at": self.started_at.isoformat() if self.started_at else None,
                        "last_action_at": self.last_action_at.isoformat() if self.last_action_at else None,
                        "seed": self.seed,
                    },
                }
            )
            return payload

    def _choose_task(self, seed: int | None) -> Task:
        if self.custom_task is not None:
            return self.custom_task
        return random.Random(seed).choice(list(TASKS.values()))

    def _issues_remaining(self) -> int:
        total = self.task_state.get("total_issues", 0)
        fixed = len(self.task_state.get("fixed_issues", set()))
        return max(0, total - fixed)

    def _evaluate_action(self, action: Action) -> dict[str, Any]:
        issues: list[dict[str, Any]] = self.current_task.config.get("issues", [])
        traps: set[int] = self.task_state["traps"]
        fixed: set[tuple[int, str]] = self.task_state["fixed_issues"]

        row = action.row_index
        col = action.column
        atype = action.action_type
        normalized_col = str(col).strip().lower()

        trap_penalty = row in traps and normalized_col == "reading"
        issues_fixed_this_step = 0
        redundant = False
        matched_issue: dict[str, Any] | None = None

        for issue in issues:
            issue_col = str(issue["column"]).strip()
            key = (issue["row_index"], issue_col)
            if issue["row_index"] == row and issue_col.lower() == normalized_col:
                if key in fixed:
                    redundant = True
                else:
                    appropriate_types = {
                        "missing": {"fill_missing", "fix_value"},
                        "string_null": {"fill_missing", "fix_value"},
                        "format": {"standardize_format", "fix_value"},
                        "case": {"standardize_format", "fix_value"},
                        "duplicate": {"drop_row"},
                        "outlier": {"flag_anomaly", "fix_value", "drop_row"},
                        "negative": {"fix_value"},
                        "invalid_fk": {"flag_anomaly", "fix_value", "drop_row"},
                        "zero_price": {"fix_value"},
                        "invalid_date": {"fix_value", "flag_anomaly"},
                        "null": {"fill_missing", "fix_value", "flag_anomaly"},
                        "pattern": {"extract_pattern", "replace_regex"},
                        "categorical": {"merge_similar_categories"},
                        "imputation": {"knn_impute", "mean_impute"},
                    }.get(issue["issue_type"], {"fix_value", "flag_anomaly"})

                    if atype in appropriate_types and self._value_matches_issue(issue, action.new_value):
                        fixed.add(key)
                        issues_fixed_this_step += 1
                        matched_issue = issue
                        self._apply_action_to_dataset(issue, action)
                        break

        if trap_penalty:
            self.task_state["penalized_traps"].add(row)

        total = self.task_state["total_issues"]
        solved = len(fixed) >= total

        if trap_penalty:
            feedback = f"Row {row} contains a valid reading that should not be modified."
        elif redundant:
            feedback = f"Row {row}, column '{col}' was already fixed. No change made."
        elif issues_fixed_this_step > 0 and matched_issue is not None:
            feedback = f"Correct — row {row}, column '{matched_issue['column']}' issue resolved ({matched_issue['issue_type']})."
        else:
            feedback = f"Row {row}, column '{col}' does not match a known issue. Check the dataset again."

        return {
            "issues_fixed_this_step": issues_fixed_this_step,
            "issues_remaining": self._issues_remaining(),
            "trap_penalty": trap_penalty,
            "redundant_action": redundant,
            "solved": solved,
            "feedback": feedback,
        }

    def _value_matches_issue(self, issue: dict[str, Any], new_value: Any) -> bool:
        if "allowed_values" in issue:
            return str(new_value).strip() in {str(value).strip() for value in issue["allowed_values"]}

        if issue.get("value_type") == "numeric_range":
            try:
                numeric_value = float(new_value)
            except (TypeError, ValueError):
                return False
            minimum = float(issue.get("min", numeric_value))
            maximum = float(issue.get("max", numeric_value))
            return minimum <= numeric_value <= maximum

        if issue.get("value_type") == "iso_date":
            return isinstance(new_value, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}$", new_value.strip()))

        if issue.get("correct") is not None:
            return str(new_value).strip() == str(issue["correct"]).strip()

        return True

    def _apply_action_to_dataset(self, issue: dict[str, Any], action: Action) -> None:
        headers: list[str] = self.task_state.get("dataset_headers", [])
        rows: list[dict[str, Any]] = self.task_state.get("dataset_rows", [])
        if not headers or not rows:
            return

        row_index = issue["row_index"]
        if row_index < 0 or row_index >= len(rows):
            return

        row = rows[row_index]
        column = str(issue["column"]).strip()
        action_type = action.action_type
        current_value = str(row.get(column, ""))

        if action_type == "drop_row":
            row["__row_status__"] = "dropped"
            return

        if action_type == "flag_anomaly":
            if "[FLAGGED]" not in current_value:
                row[column] = f"{current_value} [FLAGGED]"
            return

        if action.new_value is None:
            if issue["issue_type"] in {"missing", "string_null", "null"}:
                row[column] = "<FILLED>"
            return

        row[column] = str(action.new_value)

    def _build_observation(self) -> Observation:
        task = self.current_task
        if task is None:
            raise RuntimeError("Environment not reset.")

        attempts_remaining = int(self.task_state.get("attempts_remaining", task.max_steps))
        dataset_preview = _render_dataset_preview(
            self.task_state.get("dataset_headers", []),
            self.task_state.get("dataset_rows", []),
        ) or task.config["dataset_preview"]
        context = (
            "Episode complete."
            if self.done
            else "Review the dataset, identify the next issue, and submit one action."
        )
        return Observation(
            done=self.done,
            task_id=task.id,
            task_name=task.name,
            task_description=task.description.strip(),
            dataset_preview=dataset_preview,
            issues_remaining=self._issues_remaining(),
            step=self.step_count,
            max_steps=task.max_steps,
            attempts_remaining=attempts_remaining,
            feedback=str(self.task_state.get("feedback", "")),
            feedback_history=list(self.task_state.get("feedback_history", [])),
            available_actions=list(task.config.get("available_actions", [])) if not self.done else [],
            context=context,
        )

    def _reward_state(self, transition: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "task_id": self.current_task.id if self.current_task else None,
            "step_count": self.step_count,
            "max_steps": self.current_task.max_steps if self.current_task else 0,
            "done": self.done,
            "last_transition": transition or {},
        }

    def _build_info(self, transition: dict[str, Any]) -> dict[str, Any]:
        return {
            "attempts_remaining": self.task_state.get("attempts_remaining", 0),
            "episode_progress": round(self.step_count / max(1, self.current_task.max_steps), 3),
            "solved": transition["solved"],
            "feedback": transition["feedback"],
            "issues_fixed_this_step": transition["issues_fixed_this_step"],
            "issues_remaining": transition["issues_remaining"],
            "trap_penalty": transition.get("trap_penalty", False),
        }
