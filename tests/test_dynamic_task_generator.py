"""Tests for session-local dynamic task generation from arbitrary CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from environment import Action, OpenEnv
from environment.dynamic_task_generator import generate_task_and_grader_from_csv, generate_task_from_csv


def _write_dynamic_csv(path: Path) -> None:
    rows = [
        {"order_id": 1, "price": 120.0, "city": "Paris", "signup_date": "2024-01-15"},
        {"order_id": 2, "price": "", "city": "Berlin", "signup_date": "01/16/2024"},
        {"order_id": 3, "price": -50.0, "city": "NULL", "signup_date": "2024-01-17"},
        {"order_id": 4, "price": 9999.0, "city": "Rome", "signup_date": "2024-01-18"},
        {"order_id": 4, "price": 9999.0, "city": "Rome", "signup_date": "2024-01-18"},
        {"order_id": 5, "price": 140.0, "city": "Paris", "signup_date": "2024-01-19"},
        {"order_id": 6, "price": 160.0, "city": "Berlin", "signup_date": "2024-01-20"},
        {"order_id": 7, "price": 155.0, "city": "Rome", "signup_date": "2024-01-21"},
        {"order_id": 8, "price": 148.0, "city": "Paris", "signup_date": "2024-01-22"},
        {"order_id": 9, "price": 151.0, "city": "Berlin", "signup_date": "2024-01-23"},
        {"order_id": 10, "price": 149.0, "city": "Rome", "signup_date": "2024-01-24"},
        {"order_id": 11, "price": 152.0, "city": "Paris", "signup_date": "2024-01-25"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _action_for_issue(issue: dict[str, object]) -> dict[str, object]:
    issue_type = str(issue["issue_type"])
    action_type = {
        "missing": "fill_missing",
        "null": "fill_missing",
        "string_null": "fix_value",
        "duplicate": "drop_row",
        "outlier": "flag_anomaly",
        "negative": "fix_value",
        "format": "standardize_format",
    }.get(issue_type, "fix_value")

    if issue_type == "duplicate":
        new_value = None
    elif issue.get("correct") is not None:
        new_value = issue["correct"]
    elif issue.get("allowed_values"):
        new_value = issue["allowed_values"][0]
    elif issue.get("value_type") == "numeric_range":
        minimum = float(issue["min"])
        maximum = float(issue["max"])
        new_value = round((minimum + maximum) / 2.0, 4)
    elif issue.get("value_type") == "iso_date":
        new_value = "2024-01-16"
    else:
        new_value = "fixed"

    return {
        "action_type": action_type,
        "row_index": int(issue["row_index"]),
        "column": str(issue["column"]),
        "new_value": new_value,
        "reason": f"Resolve {issue_type} issue.",
    }


def test_generate_task_from_csv_returns_dynamic_task(tmp_path: Path) -> None:
    csv_path = tmp_path / "dynamic.csv"
    _write_dynamic_csv(csv_path)

    task = generate_task_from_csv(str(csv_path), max_issues=6, max_preview_rows=8)

    assert task.id.startswith("dynamic_dynamic_")
    assert task.config["dynamic"] is True
    assert task.config["source_file"] == str(csv_path.resolve())
    assert task.max_steps > len(task.config["issues"])
    assert task.config["issues"]
    assert "dataset_preview" in task.config


def test_dynamic_task_grader_scores_matching_action(tmp_path: Path) -> None:
    csv_path = tmp_path / "dynamic.csv"
    _write_dynamic_csv(csv_path)

    task, grader = generate_task_and_grader_from_csv(str(csv_path), max_issues=6)
    action = _action_for_issue(task.config["issues"][0])

    assert grader([]) == 0.0
    assert grader([{"action": action}]) > 0.0


def test_dynamic_env_accepts_iso_date_fix(tmp_path: Path) -> None:
    csv_path = tmp_path / "dynamic.csv"
    _write_dynamic_csv(csv_path)

    task, grader = generate_task_and_grader_from_csv(str(csv_path), max_issues=6)
    format_issue = next(issue for issue in task.config["issues"] if issue["issue_type"] == "format")

    env = OpenEnv()
    env.set_dynamic_task(task, grader)
    env.reset(task.id)
    observation, reward, done, info = env.step(Action(**_action_for_issue(format_issue)))

    assert reward.value > 0.0
    assert observation.dataset_preview != task.config["dataset_preview"]
    assert "2024-01-16" in observation.dataset_preview
    assert info["issues_fixed_this_step"] == 1
