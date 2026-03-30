"""Deterministic graders for each code review task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .tasks import get_task


GraderFunction = Callable[[list[dict[str, Any]]], float]


def score_bug_report(
    action: Mapping[str, Any],
    answer: Mapping[str, Any],
    step_number: int,
) -> tuple[float, float, dict[str, float]]:
    """Score a single attempt using the environment's exact reward rubric."""
    bug_line = action.get("bug_line")
    bug_type = action.get("bug_type")
    explanation = str(action.get("explanation") or "").lower()

    line_score = 0.0
    type_score = 0.0
    explanation_score = 0.0
    bonus_score = 0.0

    if bug_line == answer["bug_line"]:
        line_score = 0.4
    elif isinstance(bug_line, int) and abs(bug_line - answer["bug_line"]) <= 1:
        line_score = 0.15

    if bug_type == answer["bug_type"]:
        type_score = 0.3

    if any(term in explanation for term in answer["key_terms"]):
        explanation_score = 0.3

    partial_credit = min(1.0, line_score + type_score + explanation_score)
    total = partial_credit

    if total >= 1.0 and step_number == 1:
        bonus_score = 0.3
        total += bonus_score
    elif total >= 1.0 and step_number == 2:
        bonus_score = 0.1
        total += bonus_score

    total = max(0.0, min(1.0, total))
    return total, partial_credit, {
        "line_score": line_score,
        "type_score": type_score,
        "explanation_score": explanation_score,
        "bonus_score": bonus_score,
    }


def _grade_code_review(task_id: str, episode_history: list[dict[str, Any]]) -> float:
    if not episode_history:
        return 0.0

    answer = get_task(task_id).config["answer"]
    best_score = 0.0

    for step_number, step in enumerate(episode_history, start=1):
        action = step.get("action", {})
        score, _, _ = score_bug_report(action, answer, step_number)
        best_score = max(best_score, score)

    return best_score


def _grade_runtime_bug(episode_history: list[dict[str, Any]]) -> float:
    return _grade_code_review("runtime_bug", episode_history)


def _grade_binary_search_logic(episode_history: list[dict[str, Any]]) -> float:
    return _grade_code_review("binary_search_logic", episode_history)


def _grade_security_vulnerability(episode_history: list[dict[str, Any]]) -> float:
    return _grade_code_review("security_vulnerability", episode_history)


GRADERS: dict[str, GraderFunction] = {
    "runtime_bug": _grade_runtime_bug,
    "binary_search_logic": _grade_binary_search_logic,
    "security_vulnerability": _grade_security_vulnerability,
}


def get_grader(task_id: str) -> GraderFunction:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task: {task_id}")
    return GRADERS[task_id]


def grade_episode(task_id: str, episode_history: list[dict[str, Any]]) -> float:
    grader = get_grader(task_id)
    score = float(grader(episode_history))
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Grader returned invalid score: {score}")
    return score
