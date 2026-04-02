"""Tests for the EDA agent."""

from __future__ import annotations

import pandas as pd

from environment.eda_agent import EDAAgent


def test_eda_agent_offline_generates_report_and_steps() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "signup_date": ["2024-01-01", "2024-01-02", None, "2024-01-04"],
            "revenue": [10.0, 20.0, 30.0, 1000.0],
            "segment": ["A", "B", "A", None],
            "target": [1, 0, 1, 0],
        }
    )

    report = EDAAgent(df, target_column="target", use_llm=False).run()
    engineered = report.apply(df)

    assert report.task_type == "classification"
    assert len(report.column_profiles) == len(df.columns)
    assert len(report.feature_engineering_steps) >= 1
    assert engineered.shape[1] >= df.shape[1]
    assert any("__eda_missing" in column or "__year" in column or "__log1p" in column for column in engineered.columns)
