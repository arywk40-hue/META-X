"""Tests for dataset evaluation and prepare-and-evaluate workflow."""

from __future__ import annotations

import csv
from pathlib import Path

from environment.evaluation import prepare_and_evaluate_dataset


def _write_classification_csv(path: Path) -> None:
    rows = [
        {"id": 1, "target": 1, "age": 22, "fare": 10.0, "sex": "female", "port": "S"},
        {"id": 2, "target": 0, "age": 35, "fare": 8.0, "sex": "male", "port": "Q"},
        {"id": 3, "target": 1, "age": 28, "fare": 75.0, "sex": "female", "port": "C"},
        {"id": 4, "target": 0, "age": "", "fare": 7.0, "sex": "male", "port": "S"},
        {"id": 5, "target": 1, "age": 31, "fare": 50.0, "sex": "female", "port": "C"},
        {"id": 6, "target": 0, "age": 40, "fare": 9.0, "sex": "male", "port": "Q"},
        {"id": 7, "target": 1, "age": 19, "fare": 85.0, "sex": "female", "port": "S"},
        {"id": 8, "target": 0, "age": 45, "fare": 6.0, "sex": "male", "port": "Q"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_and_evaluate_dataset_returns_best_model(tmp_path: Path) -> None:
    csv_path = tmp_path / "classification.csv"
    output_dir = tmp_path / "prepared_eval"
    _write_classification_csv(csv_path)

    result = prepare_and_evaluate_dataset(
        csv_path=str(csv_path),
        target_column="target",
        output_dir=str(output_dir),
        validation_fraction=0.25,
        random_seed=7,
    )

    assert result["best_model"]
    assert result["primary_metric_name"] in {"accuracy", "roc_auc", "r2"}
    assert Path(result["evaluation"]["evaluation_report_path"]).exists()
    assert len(result["evaluation"]["leaderboard"]) >= 2


def test_prepare_and_evaluate_endpoint(test_client, tmp_path: Path) -> None:
    csv_path = tmp_path / "classification.csv"
    _write_classification_csv(csv_path)

    response = test_client.post(
        "/prepare-and-evaluate",
        json={
            "csv_path": str(csv_path),
            "target_column": "target",
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["best_model"]
    assert payload["evaluation"]["leaderboard"]
