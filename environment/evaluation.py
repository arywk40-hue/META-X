"""Model evaluation utilities for prepared tabular datasets.

Models are fitted on training rows and scored only on held-out rows. The
selection metric is explicit: if it cannot be computed (for example ROC-AUC on
a validation split with a single class) evaluation stops with an error instead
of quietly switching to another metric. Failed models are reported as failed
and never receive a score.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import numpy as np
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
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_prep import (
    DatasetPreparationArtifacts,
    load_raw_frame,
    parse_time_column,
    prepare_dataset,
    prepare_split,
    resolve_split_strategy,
)
from .reporting import write_report_bundle


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# metric name -> greater_is_better
CLASSIFICATION_METRICS = {"roc_auc": True, "accuracy": True, "f1_macro": True, "log_loss": False}
REGRESSION_METRICS = {"r2": True, "rmse": False, "mae": False}


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
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
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


def model_factory(task_type: str) -> Callable[[int], dict[str, Any]]:
    return _classification_models if task_type == "classification" else _regression_models


def resolve_scoring(scoring: str | None, task_type: str, n_classes: int, mode: str = "general") -> str:
    scoring = (scoring or "auto").strip().lower()
    if mode == "credit_scoring":
        if scoring not in {"auto", "roc_auc"}:
            raise ValueError("credit_scoring mode selects models by roc_auc only.")
        return "roc_auc"
    metrics = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    if scoring == "auto":
        if task_type == "classification":
            return "roc_auc" if n_classes == 2 else "accuracy"
        return "r2"
    if scoring not in metrics:
        raise ValueError(f"Scoring '{scoring}' is not valid for {task_type}; choose one of {sorted(metrics)}.")
    if scoring == "roc_auc" and n_classes != 2:
        raise ValueError(f"roc_auc scoring needs a binary target; this target has {n_classes} classes.")
    return scoring


def greater_is_better(scoring: str) -> bool:
    return {**CLASSIFICATION_METRICS, **REGRESSION_METRICS}[scoring]


def check_validation_split(y_train: pd.Series, y_valid: pd.Series, task_type: str, scoring: str) -> None:
    if len(y_valid) == 0:
        raise ValueError("Invalid validation split: no held-out rows.")
    if task_type != "classification":
        return
    if y_train.nunique() < 2:
        raise ValueError("Invalid validation split: training rows contain a single class.")
    if scoring in {"roc_auc", "log_loss"} and y_valid.nunique() < 2:
        raise ValueError(
            f"Invalid validation split: {scoring} needs both classes in the validation rows, "
            f"but only class {y_valid.iloc[0]!r} is present. Use a stratified split or a larger validation fraction."
        )


def _score_classifier(model: Any, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_valid)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "f1_macro": float(f1_score(y_valid, predictions, average="macro")),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_valid)
        classes = list(model.classes_)
        if len(set(y_valid)) >= 2:
            metrics["log_loss"] = float(log_loss(y_valid, probabilities, labels=classes))
        if len(classes) == 2 and len(set(y_valid)) == 2:
            # Positive class is the larger label: 1 = default in credit_scoring mode.
            positive_index = classes.index(max(classes))
            metrics["roc_auc"] = float(roc_auc_score(y_valid, probabilities[:, positive_index]))
    return metrics


def _score_regressor(model: Any, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_valid)
    return {
        "r2": float(r2_score(y_valid, predictions)),
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "rmse": float(mean_squared_error(y_valid, predictions) ** 0.5),
    }


def score_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    task_type: str,
    scoring: str,
    random_seed: int = 42,
    models: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit each model on training rows, score on validation rows. Returns (leaderboard, failed)."""
    models = models if models is not None else model_factory(task_type)(random_seed)
    scorer = _score_classifier if task_type == "classification" else _score_regressor
    leaderboard: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for model_name, model in models.items():
        start = perf_counter()
        try:
            model.fit(x_train, y_train)
            metrics = scorer(model, x_valid, y_valid)
            if scoring not in metrics or not np.isfinite(metrics[scoring]):
                raise ValueError(f"{scoring} could not be computed for this model.")
        except Exception as exc:
            failed.append({"model_name": model_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        leaderboard.append(
            {
                "model_name": model_name,
                "status": "ok",
                "training_seconds": round(perf_counter() - start, 4),
                **{key: round(value, 6) for key, value in metrics.items()},
                "primary_metric": round(metrics[scoring], 6),
                "primary_metric_name": scoring,
            }
        )
    leaderboard.sort(key=lambda item: float(item["primary_metric"]), reverse=greater_is_better(scoring))
    return leaderboard, failed


def _xy(frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in [*feature_columns, target_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"Prepared split is missing columns: {', '.join(missing[:10])}.")
    return frame[feature_columns], frame[target_column]


def evaluate_prepared_dataset(
    preparation: DatasetPreparationArtifacts,
    random_seed: int = 42,
    scoring: str = "auto",
) -> DatasetEvaluationArtifacts:
    target_column = preparation.summary.get("target_column")
    task_type = preparation.summary.get("task_type")
    if not target_column:
        raise ValueError("Evaluation requires a target column.")

    feature_columns = list(preparation.summary["feature_columns"])
    x_train, y_train = _xy(_load_frame(preparation.prepared_train_path), feature_columns, target_column)
    x_valid, y_valid = _xy(_load_frame(preparation.prepared_valid_path), feature_columns, target_column)

    n_classes = int(y_train.nunique()) if task_type == "classification" else 0
    metric = resolve_scoring(scoring, task_type, n_classes, preparation.summary.get("mode", "general"))
    check_validation_split(y_train, y_valid, task_type, metric)

    leaderboard, failed = score_models(x_train, y_train, x_valid, y_valid, task_type, metric, random_seed)
    if not leaderboard:
        reasons = "; ".join(f"{item['model_name']}: {item['error']}" for item in failed)
        raise ValueError(f"All models failed on the validation split: {reasons}")
    best_entry = leaderboard[0]

    summary = {
        "dataset_name": preparation.summary["dataset_name"],
        "task_type": task_type,
        "target_column": target_column,
        "scoring": metric,
        "greater_is_better": greater_is_better(metric),
        "primary_metric_name": metric,
        "selection_basis": "holdout validation rows (preprocessing fitted on training rows only)",
        "best_model": best_entry["model_name"],
        "best_primary_metric": best_entry["primary_metric"],
        "leaderboard": leaderboard,
        "failed_models": failed,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "prepared_train_path": str(preparation.prepared_train_path),
        "prepared_valid_path": str(preparation.prepared_valid_path),
    }

    report_path = Path(preparation.summary["output_dir"]) / f"{preparation.summary['dataset_name']}_evaluation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    return DatasetEvaluationArtifacts(summary=summary, report_path=report_path)


def _fold_indices(raw: pd.DataFrame, preparation: DatasetPreparationArtifacts, cv_folds: int) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    config = preparation.config
    task_type = preparation.summary["task_type"]
    strategy = resolve_split_strategy(config, task_type)
    positions = np.arange(len(raw))
    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_seed)
        return strategy, list(splitter.split(positions, raw[config.target_column].astype(str)))
    if strategy == "random":
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=config.random_seed)
        return strategy, list(splitter.split(positions))
    if strategy == "group":
        groups = raw[config.group_column].astype(str)
        return strategy, list(GroupKFold(n_splits=cv_folds).split(positions, groups=groups))
    # Chronological: folds are built over distinct timestamps so ties never straddle a boundary.
    times = parse_time_column(raw[config.time_column], config.time_column)
    unique_times = np.sort(times.unique())
    folds = []
    for train_units, valid_units in TimeSeriesSplit(n_splits=cv_folds).split(unique_times):
        train_mask = times.isin(unique_times[train_units]).to_numpy()
        valid_mask = times.isin(unique_times[valid_units]).to_numpy()
        folds.append((positions[train_mask], positions[valid_mask]))
    return strategy, folds


def cross_validate_dataset(
    preparation: DatasetPreparationArtifacts,
    cv_folds: int,
    scoring: str = "auto",
    random_seed: int = 42,
) -> dict[str, Any]:
    """K-fold CV over all labelled rows. Preprocessing (and EDA) is refitted inside every fold."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if preparation.config is None:
        raise ValueError("Cross-validation needs the preparation config.")
    config = preparation.config
    task_type = preparation.summary["task_type"]
    _, raw = load_raw_frame(preparation.summary["source_path"])
    n_classes = int(raw[config.target_column].nunique()) if task_type == "classification" else 0
    metric = resolve_scoring(scoring, task_type, n_classes, config.mode)
    strategy, folds = _fold_indices(raw, preparation, cv_folds)

    per_model: dict[str, dict[str, Any]] = {}
    for fold_index, (train_pos, valid_pos) in enumerate(folds, start=1):
        fold = prepare_split(raw.iloc[np.sort(train_pos)], raw.iloc[np.sort(valid_pos)], config, task_type)
        check_validation_split(fold.train.target, fold.valid.target, task_type, metric)
        leaderboard, failed = score_models(
            fold.train.features, fold.train.target, fold.valid.features, fold.valid.target, task_type, metric, random_seed
        )
        for entry in leaderboard:
            per_model.setdefault(entry["model_name"], {"fold_scores": [], "failed_folds": []})["fold_scores"].append(
                entry["primary_metric"]
            )
        for entry in failed:
            per_model.setdefault(entry["model_name"], {"fold_scores": [], "failed_folds": []})["failed_folds"].append(
                {"fold": fold_index, "error": entry["error"]}
            )

    results = []
    for model_name, record in per_model.items():
        scores = record["fold_scores"]
        results.append(
            {
                "model_name": model_name,
                "mean": round(float(np.mean(scores)), 6) if scores else None,
                "std": round(float(np.std(scores)), 6) if scores else None,
                "fold_scores": scores,
                "failed_folds": record["failed_folds"],
                # A model that failed in any fold has no comparable CV score.
                "status": "ok" if not record["failed_folds"] else "failed",
            }
        )
    ranked = sorted(
        (item for item in results if item["status"] == "ok"),
        key=lambda item: item["mean"],
        reverse=greater_is_better(metric),
    )
    return {
        "cv_folds": cv_folds,
        "strategy": strategy,
        "scoring": metric,
        "preprocessing": "refitted inside each fold on that fold's training rows",
        "results": ranked + [item for item in results if item["status"] != "ok"],
    }


def refit_final_model(preparation: DatasetPreparationArtifacts, model_name: str, random_seed: int = 42) -> dict[str, Any]:
    """Fit the chosen model on every labelled row. Not scored: there are no held-out rows left."""
    task_type = preparation.summary["task_type"]
    target_column = preparation.summary["target_column"]
    feature_columns = list(preparation.summary["full_feature_columns"])
    x_full, y_full = _xy(_load_frame(preparation.prepared_full_path), feature_columns, target_column)
    model = model_factory(task_type)(random_seed)[model_name]
    model.fit(x_full, y_full)
    model_path = Path(preparation.summary["output_dir"]) / f"{preparation.summary['dataset_name']}_final_model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
    return {
        "model_name": model_name,
        "fitted_rows": int(len(x_full)),
        "model_path": str(model_path),
        "preprocessor_path": preparation.summary["preprocessor_full_path"],
        "scored": False,
    }


def prepare_and_evaluate_dataset(
    csv_path: str,
    target_column: str,
    output_dir: str | None = None,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    use_eda_agent: bool = False,
    eda_use_llm: bool = False,
    eda_llm_strategy: str = "single_pass",
    eda_llm_rounds: int = 2,
    task_type: str = "auto",
    mode: str = "general",
    split_strategy: str = "auto",
    group_column: str | None = None,
    time_column: str | None = None,
    id_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    ratio_features: list[str] | None = None,
    scoring: str = "auto",
    cv_folds: int = 0,
    refit_final: bool = False,
) -> dict[str, Any]:
    preparation = prepare_dataset(
        csv_path=csv_path,
        target_column=target_column,
        output_dir=output_dir,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        use_eda_agent=use_eda_agent,
        eda_use_llm=eda_use_llm,
        eda_llm_strategy=eda_llm_strategy,
        eda_llm_rounds=eda_llm_rounds,
        task_type=task_type,
        mode=mode,
        split_strategy=split_strategy,
        group_column=group_column,
        time_column=time_column,
        id_columns=id_columns,
        exclude_columns=exclude_columns,
        ratio_features=ratio_features,
    )
    evaluation = evaluate_prepared_dataset(preparation, random_seed=random_seed, scoring=scoring)
    if cv_folds:
        evaluation.summary["cross_validation"] = cross_validate_dataset(
            preparation, cv_folds=cv_folds, scoring=scoring, random_seed=random_seed
        )
    if refit_final:
        evaluation.summary["final_refit"] = refit_final_model(preparation, evaluation.summary["best_model"], random_seed)
    evaluation.report_path.write_text(json.dumps(evaluation.summary, indent=2))

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
