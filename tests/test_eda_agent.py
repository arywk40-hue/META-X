"""Tests for the EDA agent."""

from __future__ import annotations

import pandas as pd

from environment import eda_agent as eda_agent_module
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


def test_eda_agent_validates_llm_steps_before_applying(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "signup_date": ["2024-01-01", "2024-01-02", None, "2024-01-04"],
            "revenue": [10.0, 20.0, 30.0, 1000.0],
            "segment": ["A", "B", "A", None],
            "target": [1, 0, 1, 0],
        }
    )

    def fake_call_llm(_payload: str):
        return (
            {
                "task_type": "classification",
                "summary": "LLM reviewed the profile.",
                "recommendations": ["Check categorical sparsity."],
                "feature_engineering_steps": [
                    {
                        "description": "Add safe revenue ratio",
                        "rationale": "Adds a simple derived feature without breaking schema.",
                        "code": (
                            "import pandas as pd\n"
                            "import numpy as np\n"
                            "if 'revenue' in df.columns:\n"
                            "    df['revenue__scaled'] = pd.to_numeric(df['revenue'], errors='coerce') / 10.0\n"
                            "return df"
                        ),
                    },
                    {
                        "description": "Unsafe file write",
                        "rationale": "This should be rejected.",
                        "code": "import os\ndf = df",
                    },
                    {
                        "description": "Drop target",
                        "rationale": "This should be rejected too.",
                        "code": "df = df.drop(columns=['target'])\ndf",
                    },
                ],
            },
            "openai-compatible",
        )

    monkeypatch.setattr(eda_agent_module, "_call_llm", fake_call_llm)

    report = EDAAgent(df, target_column="target", use_llm=True).run()
    engineered = report.apply(df)

    assert report.llm_provider == "openai-compatible"
    assert report.validated_llm_steps == 1
    assert report.rejected_llm_steps == 2
    assert len(report.llm_rejection_reasons or []) == 2
    assert any(step.description == "Add safe revenue ratio" for step in report.feature_engineering_steps)
    assert all(step.description != "Unsafe file write" for step in report.feature_engineering_steps)
    assert all(step.description != "Drop target" for step in report.feature_engineering_steps)
    assert "revenue__scaled" in engineered.columns
    assert "target" in engineered.columns


def test_eda_agent_planner_reviewer_loop_tracks_rounds_and_accepts_refined_steps(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "signup_date": ["2024-01-01", "2024-01-02", None, "2024-01-04"],
            "revenue": [10.0, 20.0, 30.0, 1000.0],
            "segment": ["A", "B", "A", None],
            "target": [1, 0, 1, 0],
        }
    )

    def fake_call_llm_custom(
        _payload: str,
        system_prompt: str,
        model_env_names=("EDA_MODEL_NAME", "MODEL_NAME"),
        anthropic_model_env="ANTHROPIC_MODEL",
    ):
        _ = model_env_names, anthropic_model_env
        if "You are the planner" in system_prompt:
            return (
                {
                    "task_type": "classification",
                    "summary": "Planner found revenue and signup timing signals.",
                    "recommendations": ["Create a scaled revenue feature."],
                    "feature_engineering_steps": [
                        {
                            "description": "Add revenue signal",
                            "rationale": "Revenue magnitude looks predictive.",
                            "code": "if 'revenue' in df.columns:\n    df['revenue__signal'] = pd.to_numeric(df['revenue'], errors='coerce') / 10.0\ndf",
                        }
                    ],
                },
                "openai-compatible",
            )
        return (
            {
                "task_type": "classification",
                "summary": "Reviewer kept the strongest planner step and added a robust date feature.",
                "recommendations": ["Retain only schema-safe feature additions."],
                "feature_engineering_steps": [
                    {
                        "description": "Add revenue signal",
                        "rationale": "Revenue magnitude looks predictive.",
                        "code": "if 'revenue' in df.columns:\n    df['revenue__signal'] = pd.to_numeric(df['revenue'], errors='coerce') / 10.0\ndf",
                    },
                    {
                        "description": "Expand signup year",
                        "rationale": "Signup timing can matter across cohorts.",
                        "code": "if 'signup_date' in df.columns:\n    parsed = pd.to_datetime(df['signup_date'], errors='coerce')\n    df['signup_date__year_signal'] = parsed.dt.year.fillna(0).astype(int)\ndf",
                    },
                ],
            },
            "openai-compatible",
        )

    monkeypatch.setattr(eda_agent_module, "_call_llm_custom", fake_call_llm_custom)

    report = EDAAgent(
        df,
        target_column="target",
        use_llm=True,
        llm_strategy="planner_reviewer",
        max_llm_rounds=2,
    ).run()
    engineered = report.apply(df)

    assert report.llm_strategy == "planner_reviewer"
    assert report.llm_rounds_run >= 1
    assert report.llm_candidate_steps >= 2
    assert report.validated_llm_steps >= 2
    assert report.rejected_llm_steps == 0
    assert len(report.llm_round_records or []) >= 1
    assert "revenue__signal" in engineered.columns
    assert "signup_date__year_signal" in engineered.columns
