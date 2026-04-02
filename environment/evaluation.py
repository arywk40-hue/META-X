"""Model evaluation utilities for prepared tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .data_prep import DatasetPreparationArtifacts, prepare_dataset
from .reporting import write_report_bundle


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@dataclass
class DatasetEvaluationArtifacts:
    summary: dict[str, Any]
    report_path: Path

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.summary)
        payload["evaluation_report_path"] = str(self.report_path)
        return payload


def _load_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Prepared dataset split not found: {path}")
    return pd.read_csv(path)


def _classification_models(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=1,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_seed),
    }


def _regression_models(random_seed: int) -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=1,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(random_state=random_seed),
    }


def _evaluate_classifier(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "f1_macro": float(f1_score(y_valid, predictions, average="macro")),
    }

    probabilities = model.predict_proba(x_valid) if hasattr(model, "predict_proba") else None
    unique_labels = sorted(pd.Series(y_valid).dropna().unique().tolist())

    if probabilities is not None and len(unique_labels) >= 2:
        try:
            metrics["log_loss"] = float(log_loss(y_valid, probabilities))
        except Exception:
            pass

    if probabilities is not None and len(unique_labels) == 2 and probabilities.shape[1] >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_valid, probabilities[:, 1]))
        except Exception:
            pass

    primary_metric = metrics.get("roc_auc", metrics["accuracy"])
    metrics["primary_metric"] = float(primary_metric)
    metrics["primary_metric_name"] = "roc_auc" if "roc_auc" in metrics else "accuracy"
    return metrics


def _evaluate_regressor(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    rmse = float(mean_squared_error(y_valid, predictions) ** 0.5)
    metrics = {
        "r2": float(r2_score(y_valid, predictions)),
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "rmse": rmse,
        "primary_metric": float(r2_score(y_valid, predictions)),
        "primary_metric_name": "r2",
    }
    return metrics


def evaluate_prepared_dataset(
    preparation: DatasetPreparationArtifacts,
    random_seed: int = 42,
) -> DatasetEvaluationArtifacts:
    target_column = preparation.summary.get("target_column")
    task_type = preparation.summary.get("task_type")
    if not target_column:
        raise ValueError("Evaluation requires a target column.")

    train_df = _load_frame(preparation.prepared_train_path)
    valid_df = _load_frame(preparation.prepared_valid_path)
    if target_column not in train_df.columns or target_column not in valid_df.columns:
        raise ValueError(f"Target column '{target_column}' missing from prepared splits.")

    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    x_valid = valid_df.drop(columns=[target_column])
    y_valid = valid_df[target_column]

    if task_type == "classification":
        models = _classification_models(random_seed)
        evaluator = _evaluate_classifier
    else:
        models = _regression_models(random_seed)
        evaluator = _evaluate_regressor

    leaderboard: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None

    for model_name, model in models.items():
        start = perf_counter()
        metrics = evaluator(model, x_train, y_train, x_valid, y_valid)
        duration = perf_counter() - start
        entry = {
            "model_name": model_name,
            "training_seconds": round(duration, 4),
            **{key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()},
        }
        leaderboard.append(entry)
        if best_entry is None or float(entry["primary_metric"]) > float(best_entry["primary_metric"]):
            best_entry = entry

    leaderboard.sort(key=lambda item: float(item["primary_metric"]), reverse=True)
    best_entry = best_entry or leaderboard[0]

    summary = {
        "dataset_name": preparation.summary["dataset_name"],
        "task_type": task_type,
        "target_column": target_column,
        "primary_metric_name": best_entry["primary_metric_name"],
        "best_model": best_entry["model_name"],
        "best_primary_metric": best_entry["primary_metric"],
        "leaderboard": leaderboard,
        "prepared_train_path": str(preparation.prepared_train_path),
        "prepared_valid_path": str(preparation.prepared_valid_path),
    }

    report_path = Path(preparation.summary["output_dir"]) / f"{preparation.summary['dataset_name']}_evaluation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    return DatasetEvaluationArtifacts(summary=summary, report_path=report_path)


def prepare_and_evaluate_dataset(
    csv_path: str,
    target_column: str,
    output_dir: str | None = None,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    use_eda_agent: bool = False,
    eda_use_llm: bool = False,
) -> dict[str, Any]:
    preparation = prepare_dataset(
        csv_path=csv_path,
        target_column=target_column,
        output_dir=output_dir,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        use_eda_agent=use_eda_agent,
        eda_use_llm=eda_use_llm,
    )
    evaluation = evaluate_prepared_dataset(preparation, random_seed=random_seed)
    profile_payload = json.loads(Path(preparation.summary["profile_path"]).read_text())
    work_queue = json.loads(Path(preparation.summary["work_queue_path"]).read_text())
    report_artifacts = write_report_bundle(
        dataset_name=preparation.summary["dataset_name"],
        output_dir=preparation.summary["output_dir"],
        source_profile=profile_payload["source_profile"],
        prepared_profile=profile_payload["prepared_profile"],
        work_queue=work_queue,
        preparation_summary=preparation.summary,
        evaluation_summary=evaluation.summary,
    )
    return {
        "preparation": preparation.as_dict(),
        "evaluation": {
            **evaluation.as_dict(),
            "markdown_report_path": report_artifacts["markdown_report_path"],
            "latex_report_path": report_artifacts["latex_report_path"],
            "graph_paths": report_artifacts["graph_paths"],
        },
        "best_model": evaluation.summary["best_model"],
        "best_primary_metric": evaluation.summary["best_primary_metric"],
        "primary_metric_name": evaluation.summary["primary_metric_name"],
    }
