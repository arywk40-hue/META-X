"""Generic dataset preparation utilities for train-ready CSV outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .eda_agent import EDAAgent, write_eda_artifacts
from .reporting import build_dataset_profile, build_work_queue, write_profile_bundle, write_report_bundle


LOW_CARDINALITY_THRESHOLD = 12
HIGH_UNIQUE_RATIO = 0.95
DATE_PARSE_THRESHOLD = 0.8
NUMERIC_PARSE_THRESHOLD = 0.9


@dataclass
class DatasetPreparationArtifacts:
    summary: dict[str, Any]
    prepared_full_path: Path
    prepared_train_path: Path | None
    prepared_valid_path: Path | None
    manifest_path: Path

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.summary)
        payload.update(
            {
                "prepared_full_path": str(self.prepared_full_path),
                "prepared_train_path": str(self.prepared_train_path) if self.prepared_train_path else None,
                "prepared_valid_path": str(self.prepared_valid_path) if self.prepared_valid_path else None,
                "manifest_path": str(self.manifest_path),
            }
        )
        return payload


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "dataset"


def _mode_or_unknown(series: pd.Series) -> Any:
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "Unknown"


def _looks_like_identifier(series: pd.Series, column_name: str) -> bool:
    lowered = column_name.lower()
    unique_ratio = float(series.nunique(dropna=False) / max(1, len(series)))
    if unique_ratio < HIGH_UNIQUE_RATIO:
        return False
    if any(token in lowered for token in ("id", "uuid", "guid", "ticket", "record", "name")):
        return True
    return False


def _coerce_numeric_like(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> pd.DataFrame:
    for column in list(df.columns):
        if column == target_column or is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            continue
        candidate = pd.to_numeric(df[column], errors="coerce")
        parse_ratio = candidate.notna().mean()
        if parse_ratio >= NUMERIC_PARSE_THRESHOLD:
            df[column] = candidate
            steps.append(f"Converted column '{column}' to numeric values.")
    return df


def _extract_datetime_features(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> pd.DataFrame:
    new_columns: dict[str, pd.Series] = {}
    to_drop: list[str] = []
    for column in list(df.columns):
        if column == target_column or is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            continue
        sample = df[column].dropna().astype(str).head(20)
        if sample.empty:
            continue
        looks_date_like = sample.str.contains(r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]{3,9}\s+\d{1,2})", regex=True).mean() >= 0.5
        if not looks_date_like:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        parse_ratio = parsed.notna().mean()
        if parse_ratio >= DATE_PARSE_THRESHOLD and parsed.nunique(dropna=True) > 1:
            base = _slugify(column)
            new_columns[f"{base}__year"] = parsed.dt.year.fillna(0).astype(int)
            new_columns[f"{base}__month"] = parsed.dt.month.fillna(0).astype(int)
            new_columns[f"{base}__day"] = parsed.dt.day.fillna(0).astype(int)
            new_columns[f"{base}__weekday"] = parsed.dt.weekday.fillna(0).astype(int)
            to_drop.append(column)
            steps.append(f"Expanded datetime column '{column}' into calendar features.")

    if to_drop:
        df = df.drop(columns=to_drop)
    for name, series in new_columns.items():
        df[name] = series
    return df


def _normalize_boolean_strings(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> pd.DataFrame:
    truthy = {"true", "yes", "y", "1"}
    falsy = {"false", "no", "n", "0"}
    for column in list(df.columns):
        if column == target_column or is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            continue
        lowered = df[column].dropna().astype(str).str.strip().str.lower()
        uniques = set(lowered.unique())
        if uniques and uniques.issubset(truthy | falsy):
            df[column] = df[column].astype(str).str.strip().str.lower().map(
                lambda value: 1 if value in truthy else 0
            )
            steps.append(f"Normalized boolean-like column '{column}' to 0/1.")
    return df


def _add_missing_indicators(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> pd.DataFrame:
    indicator_columns: list[str] = []
    for column in list(df.columns):
        if column == target_column:
            continue
        if df[column].isna().any():
            indicator_name = f"{_slugify(column)}__was_missing"
            df[indicator_name] = df[column].isna().astype(int)
            indicator_columns.append(indicator_name)
    if indicator_columns:
        steps.append(f"Added missingness indicators for {len(indicator_columns)} columns.")
    return df


def _drop_identifier_columns(
    df: pd.DataFrame,
    target_column: str | None,
    steps: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    for column in list(df.columns):
        if column == target_column:
            continue
        if _looks_like_identifier(df[column], column):
            dropped.append(column)
    if dropped:
        df = df.drop(columns=dropped)
        steps.append(f"Dropped identifier-like columns: {', '.join(dropped)}.")
    return df, dropped


def _clip_numeric_outliers(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> pd.DataFrame:
    clipped_columns: list[str] = []
    for column in df.select_dtypes(include=["number"]).columns:
        if column == target_column:
            continue
        series = df[column]
        if series.nunique(dropna=True) < 4:
            continue
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            df[column] = series.clip(lower=lower, upper=upper)
            clipped_columns.append(column)
    if clipped_columns:
        steps.append(f"Clipped extreme numeric outliers in {len(clipped_columns)} columns.")
    return df


def _engineer_text_length_features(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> pd.DataFrame:
    created: list[str] = []
    for column in list(df.columns):
        if column == target_column or is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            continue
        non_null = df[column].dropna().astype(str)
        if non_null.empty:
            continue
        average_length = non_null.str.len().mean()
        unique_ratio = non_null.nunique() / max(1, len(non_null))
        if average_length >= 10 and unique_ratio > 0.5:
            feature_name = f"{_slugify(column)}__len"
            df[feature_name] = df[column].fillna("").astype(str).str.len()
            created.append(feature_name)
    if created:
        steps.append(f"Added text-length features: {', '.join(created)}.")
    return df


def _impute_and_encode(
    df: pd.DataFrame,
    target_column: str | None,
    steps: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    encoded_columns: list[str] = []
    encoding_map: dict[str, str] = {}

    for column in list(df.columns):
        if column == target_column:
            continue
        if is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            if df[column].isna().any():
                median = float(df[column].median()) if df[column].notna().any() else 0.0
                df[column] = df[column].fillna(median)
                steps.append(f"Imputed numeric column '{column}' with median {median:.4f}.")
            continue

        unique_count = df[column].nunique(dropna=True)
        if unique_count <= LOW_CARDINALITY_THRESHOLD:
            fill_value = _mode_or_unknown(df[column])
            if df[column].isna().any():
                df[column] = df[column].fillna(fill_value)
                steps.append(f"Filled categorical column '{column}' with '{fill_value}'.")
            dummies = pd.get_dummies(df[column], prefix=_slugify(column), dtype=int)
            df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
            encoded_columns.append(column)
            encoding_map[column] = "one_hot"
            steps.append(f"One-hot encoded column '{column}' into {len(dummies.columns)} features.")
        else:
            if df[column].isna().any():
                df[column] = df[column].fillna("Unknown")
                steps.append(f"Filled high-cardinality column '{column}' with 'Unknown'.")
            frequencies = df[column].value_counts(normalize=True)
            df[f"{_slugify(column)}__freq"] = df[column].map(frequencies).fillna(0.0)
            df = df.drop(columns=[column])
            encoded_columns.append(column)
            encoding_map[column] = "frequency"
            steps.append(f"Frequency encoded high-cardinality column '{column}'.")

    return df, encoded_columns, encoding_map


def _prepare_target(df: pd.DataFrame, target_column: str | None, steps: list[str]) -> tuple[pd.DataFrame, str | None, str | None]:
    if not target_column:
        return df, None, None
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    target = df[target_column]
    task_type = "regression"
    if not is_numeric_dtype(target):
        categories = {value: idx for idx, value in enumerate(sorted(target.dropna().astype(str).unique()))}
        df[target_column] = target.astype(str).map(categories)
        task_type = "classification"
        steps.append(f"Encoded target column '{target_column}' into numeric class IDs.")
    else:
        unique_count = target.nunique(dropna=True)
        if unique_count <= 20:
            task_type = "classification"

    return df, target_column, task_type


def prepare_dataset(
    csv_path: str,
    target_column: str | None = None,
    output_dir: str | None = None,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    use_eda_agent: bool = False,
    eda_use_llm: bool = False,
) -> DatasetPreparationArtifacts:
    source_path = Path(csv_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    dataset_name = _slugify(source_path.stem)
    destination = Path(output_dir).expanduser().resolve() if output_dir else (source_path.parent / f"{dataset_name}_prepared")
    destination.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(source_path)
    raw_df.columns = [str(column).strip() for column in raw_df.columns]
    steps: list[str] = []

    source_profile = build_dataset_profile(raw_df, target_column)
    work_queue = build_work_queue(source_profile)

    eda_report = None
    eda_artifacts: dict[str, str] = {}
    df = raw_df.copy()
    if use_eda_agent:
        agent = EDAAgent(df, target_column=target_column, use_llm=eda_use_llm)
        eda_report = agent.run()
        eda_artifacts = write_eda_artifacts(eda_report, destination, dataset_name)
        df = eda_report.apply(df)
        steps.append(
            f"Applied EDA agent feature engineering with {len(eda_report.feature_engineering_steps)} generated steps."
        )

    original_shape = {"rows": int(len(df)), "columns": int(len(df.columns))}
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != original_shape["rows"]:
        steps.append(f"Removed {original_shape['rows'] - len(df)} duplicate rows.")

    df = _normalize_boolean_strings(df, target_column, steps)
    df = _coerce_numeric_like(df, target_column, steps)
    df = _extract_datetime_features(df, target_column, steps)
    df = _engineer_text_length_features(df, target_column, steps)
    df = _add_missing_indicators(df, target_column, steps)
    df, dropped_identifier_columns = _drop_identifier_columns(df, target_column, steps)
    df = _clip_numeric_outliers(df, target_column, steps)
    df, target_column, task_type = _prepare_target(df, target_column, steps)
    df, encoded_columns, encoding_map = _impute_and_encode(df, target_column, steps)

    if target_column and target_column in df.columns:
        feature_columns = [column for column in df.columns if column != target_column]
        df = df[feature_columns + [target_column]]
    else:
        feature_columns = list(df.columns)

    prepared_profile = build_dataset_profile(df, target_column)

    full_path = destination / f"{dataset_name}_prepared_full.csv"
    train_path: Path | None = None
    valid_path: Path | None = None

    df.to_csv(full_path, index=False)

    if target_column:
        valid_size = max(1, int(len(df) * validation_fraction))
        shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        valid_df = shuffled.iloc[:valid_size].reset_index(drop=True)
        train_df = shuffled.iloc[valid_size:].reset_index(drop=True)
        train_path = destination / f"{dataset_name}_prepared_train.csv"
        valid_path = destination / f"{dataset_name}_prepared_valid.csv"
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        steps.append(
            f"Split prepared dataset into train ({len(train_df)} rows) and valid ({len(valid_df)} rows)."
        )

    profile_artifacts = write_profile_bundle(
        dataset_name=dataset_name,
        output_dir=destination,
        source_profile=source_profile,
        prepared_profile=prepared_profile,
        work_queue=work_queue,
    )
    report_artifacts = write_report_bundle(
        dataset_name=dataset_name,
        output_dir=destination,
        source_profile=source_profile,
        prepared_profile=prepared_profile,
        work_queue=work_queue,
        preparation_summary={
            "steps": steps,
            "feature_count": len(feature_columns),
        },
        evaluation_summary=None,
    )

    summary = {
        "dataset_name": dataset_name,
        "source_path": str(source_path),
        "output_dir": str(destination),
        "target_column": target_column,
        "task_type": task_type,
        "original_shape": original_shape,
        "prepared_shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "dropped_identifier_columns": dropped_identifier_columns,
        "encoded_columns": encoded_columns,
        "encoding_map": encoding_map,
        "steps": steps,
        "wide_dataset_mode": source_profile["wide_dataset"],
        "source_profile_overview": {
            "rows": source_profile["rows"],
            "columns": source_profile["columns"],
            "missing_cells": source_profile["missing_cells"],
            "duplicate_rows": source_profile["duplicate_rows"],
            "top_suspicious_columns": [item["column"] for item in source_profile["top_suspicious_columns"]],
        },
        "work_queue_overview": {
            "workstream_count": len(work_queue["workstreams"]),
            "column_batch_count": len(work_queue["column_batches"]),
            "first_workstreams": [item["workstream"] for item in work_queue["workstreams"][:5]],
        },
        "profile_path": profile_artifacts["profile_path"],
        "work_queue_path": profile_artifacts["work_queue_path"],
        "markdown_report_path": report_artifacts["markdown_report_path"],
        "latex_report_path": report_artifacts["latex_report_path"],
        "graph_paths": report_artifacts["graph_paths"],
        "eda_enabled": use_eda_agent,
        "eda_used_llm": bool(eda_use_llm and eda_report is not None and eda_report.llm_provider != "none"),
        "eda_summary": eda_report.agent_summary if eda_report else None,
        "eda_recommendations": eda_report.agent_recommendations if eda_report else [],
        "eda_feature_engineering_steps": len(eda_report.feature_engineering_steps) if eda_report else 0,
        **eda_artifacts,
        "ready_for_training": True,
        "recommended_model_family": "tree_based_boosting_or_linear_baseline",
    }

    manifest_path = destination / f"{dataset_name}_feature_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))

    return DatasetPreparationArtifacts(
        summary=summary,
        prepared_full_path=full_path,
        prepared_train_path=train_path,
        prepared_valid_path=valid_path,
        manifest_path=manifest_path,
    )
