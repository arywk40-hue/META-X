"""Deterministic graders for the data-cleaning environment."""

from __future__ import annotations

from typing import Any, Callable


GraderFunction = Callable[[list[dict[str, Any]]], float]


def _actions_from_history(episode_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [step.get("action", {}) for step in episode_history]


def _grade_null_filling(episode_history: list[dict[str, Any]]) -> float:
    actions = _actions_from_history(episode_history)
    fixed = set()

    for action in actions:
        col = action.get("column", "")
        row = action.get("row_index")
        atype = action.get("action_type", "")
        val = action.get("new_value")

        if atype in ("fill_missing", "fix_value"):
            if row == 1 and col == "age" and val is not None:
                fixed.add("age_1")
            if row == 2 and col == "age" and val is not None and str(val).upper() != "NULL":
                fixed.add("age_2")
            if row == 4 and col == "email" and val is not None and "@" in str(val):
                fixed.add("email_4")

    return round(len(fixed) / 3, 4)


def _grade_format_standardization(episode_history: list[dict[str, Any]]) -> float:
    import re

    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    actions = _actions_from_history(episode_history)
    fixed = set()

    for action in actions:
        col = action.get("column", "")
        row = action.get("row_index")
        atype = action.get("action_type", "")
        val = str(action.get("new_value", "") or "")

        if atype in ("standardize_format", "fix_value"):
            if col == "date" and row in (1, 2, 4) and iso_re.match(val):
                fixed.add(f"date_{row}")
            if col == "currency" and row == 1 and val == "USD":
                fixed.add("currency_1")
            if col == "currency" and row == 3 and val == "USD":
                fixed.add("currency_3")

    return round(len(fixed) / 5, 4)


def _grade_duplicate_outlier(episode_history: list[dict[str, Any]]) -> float:
    actions = _actions_from_history(episode_history)
    fixed = set()

    for action in actions:
        col = action.get("column", "")
        row = action.get("row_index")
        atype = action.get("action_type", "")
        val = str(action.get("new_value", "") or "").lower()

        if atype == "drop_row" and row == 1:
            fixed.add("duplicate")
        if atype in ("flag_anomaly", "fix_value") and row == 2 and col == "amount":
            fixed.add("outlier")
        if atype == "fix_value" and row == 3 and col == "amount":
            try:
                if float(action.get("new_value", -1)) >= 0:
                    fixed.add("negative")
            except (TypeError, ValueError):
                pass
        if atype in ("fix_value", "standardize_format") and row == 5 and col == "status" and val == "completed":
            fixed.add("case")

    return round(len(fixed) / 4, 4)


def _grade_multi_layer(episode_history: list[dict[str, Any]]) -> float:
    import re

    actions = _actions_from_history(episode_history)
    fixed = set()

    for action in actions:
        col = action.get("column", "")
        row = action.get("row_index")
        atype = action.get("action_type", "")
        val = action.get("new_value")

        if atype == "fix_value" and row == 1 and col == "qty":
            try:
                if float(val) >= 0:
                    fixed.add("qty_1")
            except (TypeError, ValueError):
                pass
        if atype in ("flag_anomaly", "fix_value", "drop_row") and row == 2 and col == "customer_id":
            fixed.add("fk_customer")
        if atype in ("flag_anomaly", "fix_value") and row == 2 and col == "unit_price":
            try:
                if float(val or 0) > 0:
                    fixed.add("zero_price")
            except (TypeError, ValueError):
                pass
        if atype in ("flag_anomaly", "fix_value", "drop_row") and row == 3 and col == "product_id":
            fixed.add("fk_product")
        if atype in ("flag_anomaly", "fix_value") and row == 3 and col == "order_dt":
            if re.match(r"^\d{4}-\d{2}-\d{2}$", str(val or "")):
                fixed.add("bad_date")
        if atype in ("flag_anomaly", "drop_row") and row == 4 and col == "qty":
            fixed.add("qty_outlier")
        if atype == "drop_row" and row == 5:
            fixed.add("duplicate")

    base = len(fixed) / 7
    bonus = 0.1 if len(fixed) == 7 and len(actions) <= 7 else 0.0
    return round(min(1.0, base + bonus), 4)


def _grade_adversarial(episode_history: list[dict[str, Any]]) -> float:
    actions = _actions_from_history(episode_history)
    trap_rows = {0, 1, 4}
    real_issues = {(2, "reading"), (5, "reading")}
    fixed_real = set()
    penalized_traps = set()

    for action in actions:
        col = action.get("column", "")
        row = action.get("row_index")
        atype = action.get("action_type", "")

        if atype in ("fix_value", "flag_anomaly", "fill_missing"):
            if (row, col) in real_issues:
                fixed_real.add((row, col))
            if row in trap_rows and col == "reading":
                penalized_traps.add(row)

    score = len(fixed_real) * 0.5 - len(penalized_traps) * 0.3
    return round(max(0.0, min(1.0, score)), 4)


def _grade_titanic_manifest(episode_history: list[dict[str, Any]]) -> float:
    """4 issues, 0.25 each, using flexible but typed validation."""
    actions = _actions_from_history(episode_history)
    fixed = set()

    for action in actions:
        row = action.get("row_index")
        col = str(action.get("column", "")).strip().lower()
        atype = action.get("action_type", "")
        value = action.get("new_value")

        if atype not in ("fill_missing", "fix_value"):
            continue

        if row == 0 and col == "age":
            try:
                age = float(value)
            except (TypeError, ValueError):
                age = -1
            if 0 <= age <= 80:
                fixed.add("age")

        if row == 1 and col == "embarked" and str(value).strip() in {"C", "Q", "S"}:
            fixed.add("embarked_1")

        if row == 2 and col == "embarked" and str(value).strip() in {"C", "Q", "S"}:
            fixed.add("embarked_2")

        if row == 3 and col == "cabin" and str(value).strip().lower() in {"unknown", "missing"}:
            fixed.add("cabin")

    return round(len(fixed) / 4, 4)


GRADERS: dict[str, GraderFunction] = {
    "null_filling": _grade_null_filling,
    "format_standardization": _grade_format_standardization,
    "duplicate_outlier": _grade_duplicate_outlier,
    "multi_layer_pipeline": _grade_multi_layer,
    "adversarial_sensor": _grade_adversarial,
    "titanic_manifest": _grade_titanic_manifest,
}


def get_grader(task_id: str) -> GraderFunction:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task: {task_id}")
    return GRADERS[task_id]


def grade_episode(task_id: str, episode_history: list[dict[str, Any]]) -> float:
    grader = get_grader(task_id)
    score = float(grader(episode_history))
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Grader returned out-of-range score: {score}")
    return score
