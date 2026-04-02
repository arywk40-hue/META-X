"""Dynamic task generation for arbitrary CSV-backed cleaning episodes."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


GraderFunction = Callable[[list[dict[str, Any]]], float]


@dataclass
class DetectedIssue:
    row_index: int
    column: str
    issue_type: str
    description: str
    allowed_values: list[str] | None = None
    correct_value: Any = None
    value_type: str | None = None
    min_val: float | None = None
    max_val: float | None = None


def _to_issue_dict(issue: DetectedIssue) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "row_index": issue.row_index,
        "column": issue.column,
        "issue_type": issue.issue_type,
        "description": issue.description,
    }
    if issue.correct_value is not None:
        payload["correct"] = issue.correct_value
    if issue.allowed_values:
        payload["allowed_values"] = issue.allowed_values
    if issue.value_type:
        payload["value_type"] = issue.value_type
        payload["min"] = issue.min_val
        payload["max"] = issue.max_val
    return payload


def _detect_issues(df: pd.DataFrame, max_issues: int = 8) -> list[DetectedIssue]:
    issues: list[DetectedIssue] = []
    seen: set[tuple[int, str]] = set()

    def add_issue(issue: DetectedIssue) -> None:
        key = (issue.row_index, issue.column)
        if key not in seen and len(issues) < max_issues:
            seen.add(key)
            issues.append(issue)

    for column in df.columns:
        if len(issues) >= max_issues:
            break
        null_rows = df.index[df[column].isna()].tolist()
        for row in null_rows[:3]:
            if is_numeric_dtype(df[column]):
                non_null = df[column].dropna()
                if non_null.empty:
                    continue
                add_issue(
                    DetectedIssue(
                        row_index=int(row),
                        column=column,
                        issue_type="null",
                        description=f"Row {row}: '{column}' is missing and should be imputed.",
                        value_type="numeric_range",
                        min_val=round(float(non_null.quantile(0.05)), 4),
                        max_val=round(float(non_null.quantile(0.95)), 4),
                    )
                )
            else:
                mode_value = df[column].mode(dropna=True)
                allowed = [str(value) for value in df[column].dropna().unique().tolist()[:6]]
                add_issue(
                    DetectedIssue(
                        row_index=int(row),
                        column=column,
                        issue_type="missing",
                        description=f"Row {row}: '{column}' is missing.",
                        allowed_values=allowed or None,
                        correct_value=str(mode_value.iloc[0]) if not mode_value.empty else "Unknown",
                    )
                )

    string_nulls = {"null", "none", "n/a", "na", "?", "-", "unknown"}
    for column in df.select_dtypes(include="object").columns:
        if len(issues) >= max_issues:
            break
        for row in df.index:
            if len(issues) >= max_issues:
                break
            value = str(df.at[row, column]).strip().lower()
            if value in string_nulls:
                mode_value = df[column].mode(dropna=True)
                allowed_values = [
                    str(item)
                    for item in df[column].dropna().unique().tolist()
                    if str(item).strip().lower() not in string_nulls
                ][:6]
                add_issue(
                    DetectedIssue(
                        row_index=int(row),
                        column=column,
                        issue_type="string_null",
                        description=f"Row {row}: '{column}' contains literal null-like string '{df.at[row, column]}'.",
                        allowed_values=allowed_values or None,
                        correct_value=str(mode_value.iloc[0]) if not mode_value.empty else "Unknown",
                    )
                )

    duplicate_rows = df.index[df.duplicated(keep="first")].tolist()
    for row in duplicate_rows[:2]:
        if len(issues) >= max_issues:
            break
        add_issue(
            DetectedIssue(
                row_index=int(row),
                column=str(df.columns[0]),
                issue_type="duplicate",
                description=f"Row {row} is an exact duplicate of an earlier row.",
            )
        )

    for column in df.select_dtypes(include="number").columns:
        if len(issues) >= max_issues:
            break
        series = df[column].dropna()
        if len(series) < 10:
            continue
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        mask = (df[column] < q1 - 3 * iqr) | (df[column] > q3 + 3 * iqr)
        for row in df.index[mask].tolist()[:2]:
            if len(issues) >= max_issues:
                break
            add_issue(
                DetectedIssue(
                    row_index=int(row),
                    column=column,
                    issue_type="outlier",
                    description=f"Row {row}: '{column}'={df.at[row, column]} is a statistical outlier.",
                )
            )

    positive_keywords = {"price", "cost", "area", "age", "count", "salary", "amount", "quantity", "qty", "fare", "revenue"}
    for column in df.select_dtypes(include="number").columns:
        if len(issues) >= max_issues:
            break
        lowered = column.lower()
        if not any(keyword in lowered for keyword in positive_keywords):
            continue
        negative_rows = df.index[df[column] < 0].tolist()
        for row in negative_rows[:1]:
            if len(issues) >= max_issues:
                break
            upper_bound = round(float(df[column].dropna().quantile(0.95)), 4) if df[column].dropna().any() else 100.0
            add_issue(
                DetectedIssue(
                    row_index=int(row),
                    column=column,
                    issue_type="negative",
                    description=f"Row {row}: '{column}'={df.at[row, column]} is negative in a positive-only column.",
                    value_type="numeric_range",
                    min_val=0.0,
                    max_val=max(0.0, upper_bound),
                )
            )

    iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    date_keywords = {"date", "dt", "time", "created", "updated"}
    for column in df.select_dtypes(include="object").columns:
        if len(issues) >= max_issues:
            break
        if not any(keyword in column.lower() for keyword in date_keywords):
            continue
        for row in df.index:
            if len(issues) >= max_issues:
                break
            value = str(df.at[row, column]).strip()
            if value and value.lower() not in {"nan", "none"} and not iso_pattern.match(value):
                add_issue(
                    DetectedIssue(
                        row_index=int(row),
                        column=column,
                        issue_type="format",
                        description=f"Row {row}: '{column}'='{df.at[row, column]}' is not ISO-8601 format.",
                        value_type="iso_date",
                    )
                )

    return issues[:max_issues]


def _build_preview(df: pd.DataFrame, max_rows: int = 12) -> str:
    columns = df.columns.tolist()
    header = " | ".join(f"{column[:14]:<14}" for column in columns)
    separator = "-" * len(header)
    lines = [
        f"row_id | {header}",
        f"{'------'} | {separator}",
    ]
    for index, (_, row) in enumerate(df.head(max_rows).iterrows()):
        values = " | ".join(
            f"{str(row[column])[:14]:<14}" if not pd.isna(row[column]) else f"{'NULL':<14}"
            for column in columns
        )
        lines.append(f"{index:<6} | {values}")
    return "\n".join(lines)


def _infer_difficulty(issue_count: int, column_count: int, has_outliers: bool, has_formats: bool) -> str:
    score = issue_count
    if column_count > 10:
        score += 2
    if has_outliers:
        score += 1
    if has_formats:
        score += 2
    if score <= 3:
        return "easy"
    if score <= 6:
        return "medium"
    return "hard"


def generate_task_from_csv(
    csv_path: str,
    task_id: str | None = None,
    max_issues: int = 7,
    max_preview_rows: int = 12,
) -> "Task":
    from .models import Difficulty, Task

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, nrows=500)
    df.columns = [str(column).strip() for column in df.columns]
    issues = _detect_issues(df, max_issues=max_issues)
    if not issues:
        raise ValueError(f"No usable data quality issues were detected in '{path.name}'.")

    slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:6]
    resolved_task_id = task_id or f"dynamic_{slug}_{file_hash}"

    issue_types = {issue.issue_type for issue in issues}
    difficulty_name = _infer_difficulty(
        len(issues),
        len(df.columns),
        "outlier" in issue_types,
        "format" in issue_types,
    )
    difficulty = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
    }[difficulty_name]

    max_steps = len(issues) + max(3, len(issues) // 2)
    available_actions = ["fix_value", "fill_missing", "flag_anomaly"]
    if "duplicate" in issue_types:
        available_actions.append("drop_row")
    if "format" in issue_types:
        available_actions.append("standardize_format")

    success_parts: list[str] = []
    if issue_types & {"missing", "null", "string_null"}:
        success_parts.append("repair missing or null-like values")
    if "duplicate" in issue_types:
        success_parts.append("drop exact duplicates")
    if "outlier" in issue_types:
        success_parts.append("flag or correct statistical outliers")
    if "negative" in issue_types:
        success_parts.append("repair negative values in positive-only columns")
    if "format" in issue_types:
        success_parts.append("standardize date-like strings to ISO-8601")

    return Task(
        id=resolved_task_id,
        name=f"{path.stem} — Dynamic Data Cleaning ({difficulty_name})",
        description=(
            f"Auto-generated cleaning task for '{path.name}' with {len(issues)} detected issues across an unseen schema. "
            f"Fix the real data problems without relying on fixed benchmark column names."
        ),
        difficulty=difficulty,
        max_steps=max_steps,
        config={
            "dataset_preview": _build_preview(df, max_rows=max_preview_rows),
            "available_actions": available_actions,
            "issues": [_to_issue_dict(issue) for issue in issues],
            "traps": [],
            "source_file": str(path),
            "dynamic": True,
        },
        tags=["dynamic", "data-cleaning", *sorted(issue_types)],
        success_criteria="Fix all detected issues: " + "; ".join(success_parts) + ".",
    )


def _build_grader(issues: list[dict[str, Any]]) -> GraderFunction:
    issue_count = max(1, len(issues))

    def grader(episode_history: list[dict[str, Any]]) -> float:
        fixed: set[tuple[int, str]] = set()
        for step in episode_history:
            action = step.get("action", {})
            row = action.get("row_index")
            column = str(action.get("column", "")).strip()
            action_type = action.get("action_type", "")
            new_value = action.get("new_value")

            for issue in issues:
                key = (int(issue["row_index"]), str(issue["column"]).strip())
                if key in fixed:
                    continue
                if int(issue["row_index"]) != row or str(issue["column"]).strip() != column:
                    continue

                allowed_actions = {
                    "missing": {"fill_missing", "fix_value"},
                    "null": {"fill_missing", "fix_value", "flag_anomaly"},
                    "string_null": {"fill_missing", "fix_value"},
                    "duplicate": {"drop_row"},
                    "outlier": {"flag_anomaly", "fix_value", "drop_row"},
                    "negative": {"fix_value"},
                    "format": {"standardize_format", "fix_value"},
                }.get(issue["issue_type"], {"fix_value", "flag_anomaly"})
                if action_type not in allowed_actions:
                    continue

                if issue.get("allowed_values"):
                    if str(new_value).strip() not in {str(value).strip() for value in issue["allowed_values"]}:
                        continue
                elif issue.get("value_type") == "numeric_range":
                    try:
                        numeric_value = float(new_value)
                    except (TypeError, ValueError):
                        continue
                    if not (float(issue["min"]) <= numeric_value <= float(issue["max"])):
                        continue
                elif issue.get("value_type") == "iso_date":
                    if not isinstance(new_value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", new_value.strip()):
                        continue
                elif issue.get("correct") is not None:
                    if str(new_value).strip() != str(issue["correct"]).strip():
                        continue

                fixed.add(key)
                break

        return round(max(0.0, min(1.0, len(fixed) / issue_count)), 4)

    return grader


def generate_task_and_grader_from_csv(
    csv_path: str,
    task_id: str | None = None,
    max_issues: int = 7,
    max_preview_rows: int = 12,
) -> tuple["Task", GraderFunction]:
    task = generate_task_from_csv(
        csv_path=csv_path,
        task_id=task_id,
        max_issues=max_issues,
        max_preview_rows=max_preview_rows,
    )
    return task, _build_grader(task.config["issues"])
