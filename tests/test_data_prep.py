"""Tests for generic dataset preparation."""

from __future__ import annotations

import csv
from pathlib import Path

from environment.data_prep import prepare_dataset


def _write_sample_csv(path: Path) -> None:
    rows = [
        {"id": 1, "target": "yes", "age": 22, "city": "Paris", "signup_date": "2024-01-15", "notes": "first customer"},
        {"id": 2, "target": "no", "age": "", "city": "Berlin", "signup_date": "2024-01-16", "notes": "repeat buyer"},
        {"id": 3, "target": "yes", "age": 41, "city": "", "signup_date": "2024-01-17", "notes": "vip account"},
        {"id": 4, "target": "no", "age": 35, "city": "Paris", "signup_date": "2024-01-18", "notes": "new lead"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_dataset_writes_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    output_dir = tmp_path / "prepared"
    _write_sample_csv(csv_path)

    artifacts = prepare_dataset(
        csv_path=str(csv_path),
        target_column="target",
        output_dir=str(output_dir),
        validation_fraction=0.25,
        random_seed=7,
    )

    payload = artifacts.as_dict()
    assert Path(payload["prepared_full_path"]).exists()
    assert Path(payload["prepared_train_path"]).exists()
    assert Path(payload["prepared_valid_path"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert payload["target_column"] == "target"
    assert payload["ready_for_training"] is True
    assert payload["feature_count"] > 0


def test_prepare_dataset_endpoint(test_client, tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    response = test_client.post(
        "/prepare-dataset",
        json={
            "csv_path": str(csv_path),
            "target_column": "target",
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready_for_training"] is True
    assert Path(payload["prepared_full_path"]).exists()
