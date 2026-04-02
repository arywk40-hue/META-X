"""EDA and feature-engineering agent for arbitrary tabular datasets."""

from __future__ import annotations

import json
import os
import textwrap
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from pandas.api.types import is_numeric_dtype


warnings.filterwarnings("ignore")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_missing: int
    missing_pct: float
    n_unique: int
    sample_values: list[Any]
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skew: float | None = None
    q25: float | None = None
    q75: float | None = None
    top_categories: dict[str, int] | None = None


@dataclass
class CorrelationInsight:
    col_a: str
    col_b: str
    correlation: float
    insight: str


@dataclass
class FeatureEngineeringStep:
    description: str
    code: str
    rationale: str


@dataclass
class EDAReport:
    target_column: str | None
    shape: tuple[int, int]
    task_type: str
    column_profiles: list[ColumnProfile]
    correlation_insights: list[CorrelationInsight]
    data_quality_issues: list[str]
    feature_engineering_steps: list[FeatureEngineeringStep]
    agent_summary: str
    agent_recommendations: list[str]
    llm_provider: str = "none"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute proposed feature-engineering steps against a DataFrame."""
        transformed = df.copy()
        for step in self.feature_engineering_steps:
            try:
                local_ns: dict[str, Any] = {"df": transformed, "pd": pd, "np": np}
                exec(step.code, local_ns, local_ns)  # noqa: S102
                transformed = local_ns["df"]
            except Exception as exc:  # pragma: no cover - defensive path
                print(f"  [EDAAgent] Skipped step '{step.description}': {exc}")
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "shape": list(self.shape),
            "task_type": self.task_type,
            "data_quality_issues": self.data_quality_issues,
            "feature_engineering_steps": [
                {
                    "description": step.description,
                    "rationale": step.rationale,
                    "code": step.code,
                }
                for step in self.feature_engineering_steps
            ],
            "agent_summary": self.agent_summary,
            "agent_recommendations": self.agent_recommendations,
            "llm_provider": self.llm_provider,
        }


def _infer_task_type(df: pd.DataFrame, target: str | None) -> str:
    if not target or target not in df.columns:
        return "unknown"
    target_series = df[target]
    if not is_numeric_dtype(target_series):
        return "classification"
    return "classification" if target_series.nunique(dropna=True) <= 20 else "regression"


def _profile_columns(df: pd.DataFrame, target: str | None) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    for column in df.columns:
        series = df[column]
        profile = ColumnProfile(
            name=column,
            dtype=str(series.dtype),
            n_missing=int(series.isna().sum()),
            missing_pct=round(float(series.isna().mean()) * 100, 2),
            n_unique=int(series.nunique(dropna=True)),
            sample_values=series.dropna().unique().tolist()[:6],
        )
        if is_numeric_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                desc = non_null.describe()
                profile.mean = round(float(desc["mean"]), 4)
                profile.std = round(float(desc["std"]), 4)
                profile.min = round(float(desc["min"]), 4)
                profile.max = round(float(desc["max"]), 4)
                profile.skew = round(float(non_null.skew()), 4)
                profile.q25 = round(float(desc["25%"]), 4)
                profile.q75 = round(float(desc["75%"]), 4)
        else:
            value_counts = series.value_counts(dropna=True).head(8)
            profile.top_categories = {str(key): int(value) for key, value in value_counts.items()}
        profiles.append(profile)
    return profiles


def _correlation_insights(
    df: pd.DataFrame,
    target: str | None,
    threshold: float = 0.55,
) -> list[CorrelationInsight]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_columns) < 2:
        return []

    correlation_matrix = df[numeric_columns].corr()
    insights: list[CorrelationInsight] = []
    seen: set[frozenset[str]] = set()
    for index, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[index + 1 :]:
            key = frozenset({col_a, col_b})
            if key in seen:
                continue
            seen.add(key)
            coefficient = correlation_matrix.loc[col_a, col_b]
            if pd.isna(coefficient) or abs(coefficient) < threshold:
                continue
            label = "high_positive" if coefficient > 0.8 else "high_negative" if coefficient < -0.8 else "moderate"
            insights.append(
                CorrelationInsight(
                    col_a=col_a,
                    col_b=col_b,
                    correlation=round(float(coefficient), 4),
                    insight=label,
                )
            )
    return sorted(insights, key=lambda item: abs(item.correlation), reverse=True)


def _detect_quality_issues(df: pd.DataFrame, target: str | None) -> list[str]:
    issues: list[str] = []
    for column in df.columns:
        missing_ratio = float(df[column].isna().mean())
        if missing_ratio > 0.4:
            issues.append(f"Column '{column}' has {missing_ratio:.0%} missing values and may need dropping or aggressive imputation.")
        elif missing_ratio > 0.05:
            issues.append(f"Column '{column}' has {missing_ratio:.0%} missing values and should be imputed.")

    for column in df.select_dtypes(include="number").columns:
        if column == target:
            continue
        non_null = df[column].dropna()
        if len(non_null) < 2:
            continue
        coefficient_of_variation = non_null.std() / (abs(non_null.mean()) + 1e-9)
        if abs(coefficient_of_variation) < 0.01 and non_null.nunique() > 1:
            issues.append(
                f"Column '{column}' has near-zero variance (CV={coefficient_of_variation:.4f}) and may be uninformative."
            )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        issues.append(f"Dataset has {duplicate_rows} duplicate rows.")

    for column in df.select_dtypes(include="object").columns:
        if column == target:
            continue
        unique_count = int(df[column].nunique(dropna=True))
        if unique_count > 50:
            issues.append(f"Column '{column}' has {unique_count} unique values and is high-cardinality.")

    if target and target in df.columns and not is_numeric_dtype(df[target]):
        distribution = df[target].value_counts(normalize=True, dropna=True)
        if not distribution.empty and float(distribution.min()) < 0.1:
            issues.append(
                f"Target '{target}' is imbalanced with a minority class frequency of {float(distribution.min()):.1%}."
            )
    return issues


def _build_prompt_payload(
    df: pd.DataFrame,
    target: str | None,
    profiles: list[ColumnProfile],
    correlation_insights: list[CorrelationInsight],
    quality_issues: list[str],
) -> str:
    buffer = StringIO()
    buffer.write(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    if target:
        buffer.write(f"Prediction target: {target}\n")

    buffer.write("\n## Column profiles\n")
    for profile in profiles[:80]:
        buffer.write(f"\n### {profile.name} (dtype={profile.dtype}, missing={profile.missing_pct}%)\n")
        if profile.mean is not None:
            buffer.write(
                f"  numeric — mean={profile.mean}, std={profile.std}, min={profile.min}, max={profile.max}, "
                f"skew={profile.skew}, q25={profile.q25}, q75={profile.q75}\n"
            )
        else:
            categories = ", ".join(f"{key}:{value}" for key, value in (profile.top_categories or {}).items())
            buffer.write(f"  categorical — unique={profile.n_unique}, top: {categories}\n")

    if correlation_insights:
        buffer.write("\n## Notable correlations\n")
        for insight in correlation_insights[:10]:
            buffer.write(f"  {insight.col_a} <-> {insight.col_b}: r={insight.correlation} ({insight.insight})\n")

    if quality_issues:
        buffer.write("\n## Data quality issues\n")
        for issue in quality_issues:
            buffer.write(f"  - {issue}\n")

    return buffer.getvalue()


def _heuristic_feature_steps(df: pd.DataFrame, target: str | None, profiles: list[ColumnProfile]) -> list[FeatureEngineeringStep]:
    steps: list[FeatureEngineeringStep] = []

    missing_columns = [profile.name for profile in profiles if profile.name != target and profile.missing_pct > 5]
    if missing_columns:
        code = textwrap.dedent(
            f"""
            for column in {missing_columns!r}:
                try:
                    indicator_name = f"{{column}}__eda_missing"
                    if indicator_name not in df.columns:
                        df[indicator_name] = df[column].isna().astype(int)
                except Exception:
                    continue
            df
            """
        ).strip()
        steps.append(
            FeatureEngineeringStep(
                description="Add missingness indicators",
                rationale="Missingness itself often carries predictive signal for arbitrary incoming schemas.",
                code=code,
            )
        )

    date_like_columns = [
        profile.name
        for profile in profiles
        if profile.name != target and profile.dtype == "object" and any(token in profile.name.lower() for token in ("date", "time", "dt"))
    ]
    if date_like_columns:
        code = textwrap.dedent(
            f"""
            for column in {date_like_columns!r}:
                try:
                    parsed = pd.to_datetime(df[column], errors="coerce")
                    if parsed.notna().mean() >= 0.5:
                        base = column.strip().lower().replace(" ", "_")
                        df[f"{{base}}__year"] = parsed.dt.year.fillna(0).astype(int)
                        df[f"{{base}}__month"] = parsed.dt.month.fillna(0).astype(int)
                        df[f"{{base}}__day"] = parsed.dt.day.fillna(0).astype(int)
                except Exception:
                    continue
            df
            """
        ).strip()
        steps.append(
            FeatureEngineeringStep(
                description="Expand date columns",
                rationale="Date-like columns should be decomposed into calendar features for new datasets with unseen schemas.",
                code=code,
            )
        )

    skewed_positive = [
        profile.name
        for profile in profiles
        if profile.name != target and profile.skew is not None and profile.skew > 1.2 and (profile.min or 0) >= 0
    ]
    if skewed_positive:
        code = textwrap.dedent(
            f"""
            for column in {skewed_positive!r}:
                try:
                    new_name = f"{{column}}__log1p"
                    if new_name not in df.columns:
                        numeric = pd.to_numeric(df[column], errors="coerce")
                        df[new_name] = np.log1p(numeric.clip(lower=0))
                except Exception:
                    continue
            df
            """
        ).strip()
        steps.append(
            FeatureEngineeringStep(
                description="Create log-scaled numeric features",
                rationale="Positive skew is common in arbitrary real-world tabular datasets and log features are broadly useful.",
                code=code,
            )
        )

    high_cardinality_columns = [
        profile.name
        for profile in profiles
        if profile.name != target and profile.top_categories is not None and profile.n_unique > 20
    ]
    if high_cardinality_columns:
        code = textwrap.dedent(
            f"""
            for column in {high_cardinality_columns!r}:
                try:
                    freq_name = f"{{column}}__eda_freq"
                    if freq_name not in df.columns:
                        frequencies = df[column].astype(str).value_counts(normalize=True)
                        df[freq_name] = df[column].astype(str).map(frequencies).fillna(0.0)
                except Exception:
                    continue
            df
            """
        ).strip()
        steps.append(
            FeatureEngineeringStep(
                description="Add frequency encodings for high-cardinality categoricals",
                rationale="Frequency features help when new datasets arrive with many categories and no fixed column assumptions.",
                code=code,
            )
        )

    return steps[:5]


def _call_openai_compatible(payload_text: str) -> dict[str, Any]:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("EDA_MODEL_NAME") or os.getenv("MODEL_NAME")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")
    if not api_base_url or not model_name or not api_key:
        raise RuntimeError("OpenAI-compatible EDA call requires API_BASE_URL, MODEL_NAME/EDA_MODEL_NAME, and an API key.")

    system_prompt = textwrap.dedent(
        """
        You are an expert data scientist performing EDA and feature engineering.
        Analyse the dataset summary provided and respond ONLY with valid JSON.

        JSON schema:
        {
          "task_type": "regression" | "classification" | "unknown",
          "summary": "<2-4 sentence narrative about the dataset>",
          "recommendations": ["<actionable insight 1>", "..."],
          "feature_engineering_steps": [
            {
              "description": "<short name>",
              "rationale": "<why this helps>",
              "code": "<Python snippet that receives df as a pandas DataFrame and returns df>"
            }
          ]
        }

        Rules:
        - Each code snippet must be safe, self-contained, and work on arbitrary column names.
        - Do not reference external files, packages, or APIs.
        - Keep each step narrow and robust to missing columns.
        - Return 3 to 6 high-impact steps maximum.
        """
    ).strip()

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_text},
            ],
            temperature=0.2,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_text},
            ],
            temperature=0.2,
            max_tokens=1800,
        )
    content = completion.choices[0].message.content or "{}"
    cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def _call_anthropic(payload_text: str) -> dict[str, Any]:
    import urllib.request

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    system_prompt = textwrap.dedent(
        """
        You are an expert data scientist performing EDA and feature engineering.
        Analyse the dataset summary and respond ONLY with valid JSON.

        JSON schema:
        {
          "task_type": "regression" | "classification" | "unknown",
          "summary": "<2-4 sentence narrative about the dataset>",
          "recommendations": ["<actionable insight 1>", "..."],
          "feature_engineering_steps": [
            {
              "description": "<short name>",
              "rationale": "<why this helps>",
              "code": "<Python snippet that receives df as a pandas DataFrame and returns df>"
            }
          ]
        }
        """
    ).strip()

    body = json.dumps(
        {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": payload_text}],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = json.loads(response.read().decode("utf-8"))
    text = raw["content"][0]["text"].strip()
    cleaned = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def _call_llm(payload_text: str) -> tuple[dict[str, Any], str]:
    provider = os.getenv("EDA_PROVIDER", "").strip().lower()
    if provider == "anthropic":
        return _call_anthropic(payload_text), "anthropic"
    if provider == "openai":
        return _call_openai_compatible(payload_text), "openai-compatible"

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _call_anthropic(payload_text), "anthropic"
        except Exception:
            pass
    return _call_openai_compatible(payload_text), "openai-compatible"


def write_eda_artifacts(report: EDAReport, output_dir: str | Path, dataset_name: str) -> dict[str, str]:
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{dataset_name}_eda_report.json"
    markdown_path = reports_dir / f"{dataset_name}_eda_report.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    quality_lines = [f"- {issue}" for issue in report.data_quality_issues] or ["- No major issues detected."]
    recommendation_lines = [f"- {item}" for item in report.agent_recommendations] or ["- No extra recommendations."]
    markdown = [
        f"# {dataset_name} EDA agent report",
        "",
        f"- Task type: `{report.task_type}`",
        f"- Shape: `{report.shape[0]} rows x {report.shape[1]} columns`",
        f"- LLM provider: `{report.llm_provider}`",
        "",
        "## Summary",
        "",
        report.agent_summary or "No summary generated.",
        "",
        "## Data Quality Issues",
        "",
        *quality_lines,
        "",
        "## Recommendations",
        "",
        *recommendation_lines,
        "",
        "## Feature Engineering Steps",
        "",
    ]
    for index, step in enumerate(report.feature_engineering_steps, start=1):
        markdown.extend(
            [
                f"### {index}. {step.description}",
                "",
                f"- Rationale: {step.rationale}",
                "",
                "```python",
                step.code,
                "```",
                "",
            ]
        )
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    return {
        "eda_report_path": str(json_path),
        "eda_markdown_path": str(markdown_path),
    }


class EDAAgent:
    """Drop-in EDA + feature-engineering agent for arbitrary tabular datasets."""

    def __init__(self, df: pd.DataFrame, target_column: str | None = None, use_llm: bool = True) -> None:
        self.df = df.copy()
        self.target = target_column
        self.use_llm = use_llm

    def run(self) -> EDAReport:
        print("  [EDAAgent] Profiling columns…")
        profiles = _profile_columns(self.df, self.target)
        print("  [EDAAgent] Computing correlations…")
        correlation_insights = _correlation_insights(self.df, self.target)
        print("  [EDAAgent] Detecting quality issues…")
        quality_issues = _detect_quality_issues(self.df, self.target)

        feature_steps = _heuristic_feature_steps(self.df, self.target, profiles)
        agent_summary = "Statistical EDA complete. Using heuristic recommendations."
        agent_recommendations = [
            "Start with missing-value repair and schema review before feature engineering.",
            "Review high-cardinality columns in batches rather than attempting the full schema at once.",
        ]
        task_type = _infer_task_type(self.df, self.target)
        llm_provider = "none"

        if self.use_llm:
            print("  [EDAAgent] Calling LLM for deeper insights…")
            payload = _build_prompt_payload(self.df, self.target, profiles, correlation_insights, quality_issues)
            try:
                result, llm_provider = _call_llm(payload)
                task_type = result.get("task_type", task_type)
                agent_summary = result.get("summary", agent_summary)
                agent_recommendations = result.get("recommendations", agent_recommendations)
                llm_steps: list[FeatureEngineeringStep] = []
                for step in result.get("feature_engineering_steps", []):
                    llm_steps.append(
                        FeatureEngineeringStep(
                            description=step.get("description", ""),
                            rationale=step.get("rationale", ""),
                            code=step.get("code", "df"),
                        )
                    )
                if llm_steps:
                    feature_steps = llm_steps
                print(f"  [EDAAgent] LLM proposed {len(feature_steps)} feature-engineering steps.")
            except Exception as exc:
                agent_summary = f"LLM analysis unavailable ({exc}). Falling back to heuristic EDA recommendations."
                print(f"  [EDAAgent] Warning: {agent_summary}")

        return EDAReport(
            target_column=self.target,
            shape=self.df.shape,
            task_type=task_type,
            column_profiles=profiles,
            correlation_insights=correlation_insights,
            data_quality_issues=quality_issues,
            feature_engineering_steps=feature_steps,
            agent_summary=agent_summary,
            agent_recommendations=agent_recommendations,
            llm_provider=llm_provider,
        )


def main() -> None:
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    target = sys.argv[2] if len(sys.argv) > 2 else None
    use_llm = len(sys.argv) <= 3 or sys.argv[3].lower() != "false"

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        rng = np.random.default_rng(42)
        rows = 300
        df = pd.DataFrame(
            {
                "area": rng.integers(500, 5000, rows),
                "bedrooms": rng.integers(1, 6, rows),
                "age": rng.integers(0, 60, rows),
                "furnishing": rng.choice(["furnished", "semi", "unfurnished"], rows),
                "price": rng.integers(100_000, 2_000_000, rows),
            }
        )
        target = "price"
        print("  [EDAAgent] Using built-in demo dataset.")

    report = EDAAgent(df, target_column=target, use_llm=use_llm).run()
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
