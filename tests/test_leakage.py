"""Regression tests for leakage-safe preparation and trustworthy evaluation.

All data is synthetic and written to pytest's tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment import eda_agent as eda_agent_module
from environment.data_prep import (
    PreprocessorState,
    apply_preprocessor,
    build_config,
    load_raw_frame,
    prepare_dataset,
    prepare_split,
    split_rows,
    transform_features,
)
from environment.evaluation import (
    evaluate_prepared_dataset,
    prepare_and_evaluate_dataset,
    score_models,
)


def _credit_frame(rows: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(50_000, 15_000, rows).round(2)
    debt = rng.normal(20_000, 8_000, rows).round(2)
    income[::11] = np.nan
    default = (debt / np.nan_to_num(income, nan=50_000) + rng.normal(0, 0.15, rows) > 0.45).astype(int)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(rows)],
            "borrower_id": [f"B{i // 3:03d}" for i in range(rows)],
            "application_date": pd.date_range("2020-01-01", periods=rows // 2, freq="7D").repeat(2).strftime("%Y-%m-%d"),
            "income": income,
            "debt": debt,
            "region": rng.choice(["north", "south", "east", "west"], rows),
            "employer": [f"emp_{i % 40}" for i in range(rows)],
            "has_mortgage": rng.choice(["yes", "no", None], rows),
            "default_24m": default,
        }
    )


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def _split_once(df: pd.DataFrame, tmp_path: Path, **overrides):
    config = build_config(target_column="default_24m", id_columns=["customer_id"], **overrides)
    _, raw = load_raw_frame(_write(df, tmp_path / "raw.csv"))
    train_pos, valid_pos, strategy = split_rows(raw, config, "classification")
    return config, raw, train_pos, valid_pos, strategy


def test_validation_values_do_not_change_fitted_state(tmp_path: Path) -> None:
    df = _credit_frame()
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path)
    baseline = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification").state

    perturbed = raw.copy()
    rows = perturbed.index[valid_pos]
    perturbed.loc[rows, "income"] = 10_000_000.0
    perturbed.loc[rows, "debt"] = np.nan
    perturbed.loc[rows, "region"] = "never_seen_region"
    perturbed.loc[rows, "employer"] = "brand_new_employer"
    changed = prepare_split(perturbed.iloc[train_pos], perturbed.iloc[valid_pos], config, "classification")

    assert changed.state == baseline
    assert changed.unseen_categories_in_valid["region"] == len(valid_pos)


def test_validation_labels_do_not_change_prepared_features(tmp_path: Path) -> None:
    df = _credit_frame()
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path)
    baseline = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification")

    flipped = raw.copy()
    rows = flipped.index[valid_pos]
    flipped.loc[rows, "default_24m"] = 1 - flipped.loc[rows, "default_24m"]
    changed = prepare_split(flipped.iloc[train_pos], flipped.iloc[valid_pos], config, "classification")

    pd.testing.assert_frame_equal(changed.train.features, baseline.train.features)
    pd.testing.assert_frame_equal(changed.valid.features, baseline.valid.features)


def test_eda_steps_are_chosen_from_training_rows_only(tmp_path: Path) -> None:
    df = _credit_frame()
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path, use_eda_agent=True)
    baseline = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification")

    perturbed = raw.copy()
    perturbed.loc[perturbed.index[valid_pos], "debt"] = np.nan
    changed = prepare_split(perturbed.iloc[train_pos], perturbed.iloc[valid_pos], config, "classification")
    assert changed.state == baseline.state


def test_group_split_keeps_borrowers_together(tmp_path: Path) -> None:
    _, raw, train_pos, valid_pos, strategy = _split_once(_credit_frame(), tmp_path, group_column="borrower_id")
    assert strategy == "group"
    train_groups = set(raw.iloc[train_pos]["borrower_id"])
    valid_groups = set(raw.iloc[valid_pos]["borrower_id"])
    assert train_groups and valid_groups
    assert not train_groups & valid_groups


def test_time_split_is_chronological_and_keeps_ties_together(tmp_path: Path) -> None:
    _, raw, train_pos, valid_pos, strategy = _split_once(_credit_frame(), tmp_path, time_column="application_date")
    assert strategy == "time"
    train_times = pd.to_datetime(raw.iloc[train_pos]["application_date"])
    valid_times = pd.to_datetime(raw.iloc[valid_pos]["application_date"])
    assert train_times.max() < valid_times.min()
    assert not set(train_times) & set(valid_times)


@pytest.mark.parametrize("fraction,expected_valid", [(0.2, 24), (0.4, 48)])
def test_stratified_split_sizes_and_class_balance(tmp_path: Path, fraction: float, expected_valid: int) -> None:
    df = _credit_frame()
    _, raw, train_pos, valid_pos, strategy = _split_once(df, tmp_path, validation_fraction=fraction)
    assert strategy == "stratified"
    assert len(valid_pos) == expected_valid
    overall = raw["default_24m"].mean()
    assert abs(raw.iloc[valid_pos]["default_24m"].mean() - overall) < 0.05


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"group_column": "missing_col"}, "group_column references columns not in the dataset"),
        ({"split_strategy": "group"}, "requires group_column"),
        ({"split_strategy": "time"}, "requires time_column"),
        ({"validation_fraction": 1.5}, "validation_fraction"),
        ({"ratio_features": ["debt"]}, "numerator_column:denominator_column"),
        ({"id_columns": ["default_24m"]}, "cannot also be an id"),
    ],
)
def test_invalid_configurations_are_rejected(tmp_path: Path, overrides: dict, message: str) -> None:
    csv_path = _write(_credit_frame(), tmp_path / "train.csv")
    kwargs = {"target_column": "default_24m", "output_dir": str(tmp_path / "out"), **overrides}
    with pytest.raises(ValueError, match=message):
        prepare_dataset(csv_path=str(csv_path), **kwargs)


def test_credit_mode_requires_binary_target(tmp_path: Path) -> None:
    df = _credit_frame()
    df["default_24m"] = df["default_24m"].replace({1: 2})
    csv_path = _write(df, tmp_path / "train.csv")
    with pytest.raises(ValueError, match="binary 0/1 target"):
        prepare_dataset(csv_path=str(csv_path), target_column="default_24m", mode="credit_scoring")


def test_unseen_categories_and_missing_columns_at_transform(tmp_path: Path) -> None:
    df = _credit_frame()
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path)
    state = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification").state

    new_rows = raw.iloc[valid_pos[:3]].drop(columns=["source_row", "customer_id", "default_24m"]).copy()
    new_rows["region"] = "mars"
    new_rows["employer"] = "unknown_corp"
    features, unseen = transform_features(new_rows, state)
    region_columns = [column for column in state.feature_columns if column.startswith("region_")]
    assert (features[region_columns] == 0).all().all()
    assert (features["employer__freq"] == 0).all()
    assert unseen == {"region": 3, "employer": 3}

    with pytest.raises(ValueError, match="missing columns required"):
        transform_features(new_rows.drop(columns=["income"]), state)


def test_all_missing_feature_is_dropped_explicitly(tmp_path: Path) -> None:
    df = _credit_frame()
    df["empty_col"] = np.nan
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path)
    result = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification")
    assert result.state.all_missing_columns == ["empty_col"]
    assert any("no training values" in step for step in result.steps)


def test_missing_booleans_stay_missing(tmp_path: Path) -> None:
    df = _credit_frame()
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path)
    state = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification").state
    assert "has_mortgage" in state.boolean_columns
    sample = pd.DataFrame({column: raw[column].iloc[:3] for column in state.input_columns})
    sample["has_mortgage"] = ["yes", "no", None]
    features, _ = transform_features(sample, state)
    assert features["has_mortgage__was_missing"].tolist() == [0, 0, 1]
    assert features["has_mortgage"].iloc[0] == 1.0
    assert features["has_mortgage"].iloc[1] == 0.0
    # The missing value is imputed from the training median, not forced to False,
    # and the indicator records that it was missing.
    assert features["has_mortgage"].iloc[2] == state.numeric_fill["has_mortgage"]


def test_ids_preserved_outside_features_and_rows_never_dropped(tmp_path: Path) -> None:
    df = _credit_frame()
    df = pd.concat([df, df.iloc[[5, 5, 6]]], ignore_index=True)  # duplicates
    csv_path = _write(df, tmp_path / "train.csv")
    artifacts = prepare_dataset(
        csv_path=str(csv_path),
        target_column="default_24m",
        id_columns=["customer_id"],
        output_dir=str(tmp_path / "out"),
    )
    summary = artifacts.summary
    assert "customer_id" not in summary["feature_columns"]
    assert "source_row" not in summary["feature_columns"]
    assert "default_24m" not in summary["feature_columns"]

    train = pd.read_csv(artifacts.prepared_train_path)
    valid = pd.read_csv(artifacts.prepared_valid_path)
    assert train["source_row"].is_monotonic_increasing
    assert valid["source_row"].is_monotonic_increasing
    assert (valid["customer_id"] == df.loc[valid["source_row"], "customer_id"].to_numpy()).all()
    removed = summary["duplicate_rows_removed_from_train"]
    assert len(train) + len(valid) + removed == len(df)
    assert set(valid["source_row"]) | set(train["source_row"]) <= set(range(len(df)))

    test_df = df.drop(columns=["default_24m"]).iloc[::-1].reset_index(drop=True)
    test_df.loc[0, "region"] = "unseen"
    test_path = _write(test_df, tmp_path / "test.csv")
    result = apply_preprocessor(str(test_path), summary["preprocessor_full_path"], str(tmp_path / "test_features.csv"))
    out = pd.read_csv(result["output_path"])
    assert len(out) == len(test_df)
    assert out["customer_id"].tolist() == test_df["customer_id"].tolist()
    assert result["unseen_categories"] == {"region": 1}


def test_save_reload_gives_identical_transform(tmp_path: Path) -> None:
    df = _credit_frame()
    df["grade"] = np.random.default_rng(1).choice([1, 2, 3], len(df))  # int categories via string keys
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path, ratio_features=["debt:income"], use_eda_agent=True)
    result = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification")
    path = result.state.save(tmp_path / "state.json")
    reloaded = PreprocessorState.load(path)
    assert reloaded == result.state

    valid_features = raw.iloc[valid_pos].drop(columns=["source_row", "customer_id", "default_24m"])
    before, _ = transform_features(valid_features, result.state)
    after, _ = transform_features(valid_features, reloaded)
    pd.testing.assert_frame_equal(before, after)
    assert "debt__per__income" in before.columns


def test_generated_code_cannot_touch_target_or_ids(tmp_path: Path, monkeypatch) -> None:
    seen_columns: list[list[str]] = []

    def fake_call_llm(payload: str):
        seen_columns.append([line for line in payload.splitlines() if line.startswith("### ")])
        return (
            {
                "summary": "fake",
                "recommendations": [],
                "feature_engineering_steps": [
                    {"description": "leak target", "rationale": "r", "code": "df['x'] = df['default_24m'] * 2\ndf"},
                    {"description": "overwrite target", "rationale": "r", "code": "df['default_24m'] = 1\ndf"},
                    {"description": "use id", "rationale": "r", "code": "df['id_len'] = df['customer_id'].str.len()\ndf"},
                    {"description": "row mean", "rationale": "r", "code": "df['income_c'] = df['income'] - df['income'].mean()\ndf"},
                    {"description": "safe ratio", "rationale": "r", "code": "df['debt_x2'] = pd.to_numeric(df['debt']) * 2\ndf"},
                ],
            },
            "openai-compatible",
        )

    monkeypatch.setattr(eda_agent_module, "_call_llm", fake_call_llm)
    df = _credit_frame()
    config, raw, train_pos, valid_pos, _ = _split_once(df, tmp_path, use_eda_agent=True, eda_use_llm=True)
    result = prepare_split(raw.iloc[train_pos], raw.iloc[valid_pos], config, "classification")

    payload_columns = " ".join(seen_columns[0])
    assert "default_24m" not in payload_columns
    assert "customer_id" not in payload_columns
    accepted = {step.description for step in result.eda_steps}
    assert "safe ratio" in accepted
    assert not accepted & {"leak target", "overwrite target", "use id", "row mean"}
    assert result.eda_report.rejected_llm_steps == 4
    pd.testing.assert_series_equal(
        result.valid.target.reset_index(drop=True),
        raw.iloc[valid_pos]["default_24m"].reset_index(drop=True),
    )


def test_roc_auc_strict_and_single_class_validation_is_flagged(tmp_path: Path) -> None:
    df = _credit_frame()
    csv_path = _write(df, tmp_path / "train.csv")
    artifacts = prepare_dataset(
        csv_path=str(csv_path),
        target_column="default_24m",
        mode="credit_scoring",
        id_columns=["customer_id"],
        output_dir=str(tmp_path / "out"),
    )
    evaluation = evaluate_prepared_dataset(artifacts)
    assert evaluation.summary["primary_metric_name"] == "roc_auc"
    assert all(entry["primary_metric_name"] == "roc_auc" for entry in evaluation.summary["leaderboard"])

    valid = pd.read_csv(artifacts.prepared_valid_path)
    valid["default_24m"] = 0
    valid.to_csv(artifacts.prepared_valid_path, index=False)
    with pytest.raises(ValueError, match="Invalid validation split"):
        evaluate_prepared_dataset(artifacts)

    with pytest.raises(ValueError, match="roc_auc only"):
        evaluate_prepared_dataset(artifacts, scoring="accuracy")


def test_failed_models_are_reported_without_scores() -> None:
    class Broken:
        def fit(self, x, y):
            raise RuntimeError("boom")

    from sklearn.linear_model import LogisticRegression

    x = pd.DataFrame({"a": np.arange(20, dtype=float)})
    y = pd.Series([0, 1] * 10)
    leaderboard, failed = score_models(
        x, y, x, y, "classification", "roc_auc", models={"broken": Broken(), "lr": LogisticRegression()}
    )
    assert [entry["model_name"] for entry in leaderboard] == ["lr"]
    assert failed == [{"model_name": "broken", "status": "failed", "error": "RuntimeError: boom"}]
    assert "primary_metric" not in failed[0]


def test_lower_is_better_metric_sorts_ascending(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(size=80), "noise": rng.normal(size=80)})
    df["y"] = df["x"] * 3 + rng.normal(scale=0.1, size=80) + 100
    csv_path = _write(df, tmp_path / "reg.csv")
    result = prepare_and_evaluate_dataset(
        csv_path=str(csv_path), target_column="y", task_type="regression", scoring="rmse", output_dir=str(tmp_path / "o")
    )
    scores = [entry["rmse"] for entry in result["evaluation"]["leaderboard"]]
    assert scores == sorted(scores)
    assert result["evaluation"]["greater_is_better"] is False


@pytest.mark.parametrize("extra", [{}, {"group_column": "borrower_id"}, {"time_column": "application_date"}])
def test_cross_validation_refits_per_fold_and_final_refit_is_separate(tmp_path: Path, extra: dict) -> None:
    csv_path = _write(_credit_frame(rows=150), tmp_path / "train.csv")
    result = prepare_and_evaluate_dataset(
        csv_path=str(csv_path),
        target_column="default_24m",
        mode="credit_scoring",
        id_columns=["customer_id"],
        cv_folds=3,
        refit_final=True,
        output_dir=str(tmp_path / "out"),
        **extra,
    )
    cv = result["evaluation"]["cross_validation"]
    assert cv["scoring"] == "roc_auc"
    assert all(len(item["fold_scores"]) == 3 for item in cv["results"] if item["status"] == "ok")
    final = result["evaluation"]["final_refit"]
    assert final["scored"] is False
    assert final["fitted_rows"] == 150
    assert Path(final["model_path"]).exists()
    report = json.loads(Path(result["evaluation"]["evaluation_report_path"]).read_text())
    assert "final_refit" in report and "cross_validation" in report


def test_post_outcome_columns_flagged_in_credit_mode(tmp_path: Path) -> None:
    df = _credit_frame()
    df["recoveries"] = 0.0
    csv_path = _write(df, tmp_path / "train.csv")
    artifacts = prepare_dataset(csv_path=str(csv_path), target_column="default_24m", mode="credit_scoring")
    assert artifacts.summary["possible_post_outcome_columns"] == ["recoveries"]

    excluded = prepare_dataset(
        csv_path=str(csv_path),
        target_column="default_24m",
        mode="credit_scoring",
        exclude_columns=["recoveries"],
        output_dir=str(tmp_path / "excl"),
    )
    assert "recoveries" not in excluded.summary["feature_columns"]
    assert excluded.summary["possible_post_outcome_columns"] == []
