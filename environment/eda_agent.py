"""EDA and feature-engineering agent for arbitrary tabular datasets."""

from __future__ import annotations

import ast
import json
import os
import re
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

from .local_secrets import get_runtime_secret


warnings.filterwarnings("ignore")


BLOCKED_STEP_TOKENS = {
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "subprocess",
    "requests",
    "httpx",
    "urllib",
    "read_csv(",
    "read_parquet(",
    "to_csv(",
    "to_parquet(",
    "Path(",
    "pathlib",
    "shutil",
    "os.",
    "sys.",
}


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
class LLMRoundRecord:
    round_index: int
    planner_summary: str
    planner_recommendations: list[str]
    planner_step_count: int
    reviewer_summary: str
    reviewer_recommendations: list[str]
    reviewer_step_count: int
    accepted_step_count: int
    rejected_step_count: int


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
    llm_strategy: str = "none"
    llm_rounds_run: int = 0
    llm_candidate_steps: int = 0
    validated_llm_steps: int = 0
    rejected_llm_steps: int = 0
    llm_rejection_reasons: list[str] | None = None
    llm_round_records: list[LLMRoundRecord] | None = None

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
            "llm_strategy": self.llm_strategy,
            "llm_rounds_run": self.llm_rounds_run,
            "llm_candidate_steps": self.llm_candidate_steps,
            "validated_llm_steps": self.validated_llm_steps,
            "rejected_llm_steps": self.rejected_llm_steps,
            "llm_rejection_reasons": self.llm_rejection_reasons or [],
            "llm_round_records": [
                {
                    "round_index": record.round_index,
                    "planner_summary": record.planner_summary,
                    "planner_recommendations": record.planner_recommendations,
                    "planner_step_count": record.planner_step_count,
                    "reviewer_summary": record.reviewer_summary,
                    "reviewer_recommendations": record.reviewer_recommendations,
                    "reviewer_step_count": record.reviewer_step_count,
                    "accepted_step_count": record.accepted_step_count,
                    "rejected_step_count": record.rejected_step_count,
                }
                for record in (self.llm_round_records or [])
            ],
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


BASE_LLM_RESPONSE_SCHEMA = """
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


SINGLE_PASS_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are an expert data scientist performing EDA and feature engineering.
    Analyse the dataset summary provided and respond ONLY with valid JSON.

    {BASE_LLM_RESPONSE_SCHEMA.strip()}

    Rules:
    - Each code snippet must be safe, self-contained, and work on arbitrary column names.
    - Do not reference external files, packages, shell commands, or APIs.
    - Do not import modules. Assume `pd` and `np` already exist.
    - Keep each step narrow and robust to missing columns.
    - Return 3 to 6 high-impact steps maximum.
    """
).strip()


PLANNER_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are the planner in a two-agent EDA system.
    Your job is to propose the strongest schema-grounded feature engineering plan you can.
    You should be ambitious, but every step must still be robust to unseen tabular datasets.
    Respond ONLY with valid JSON.

    {BASE_LLM_RESPONSE_SCHEMA.strip()}

    Planner rules:
    - Propose 3 to 6 candidate steps maximum.
    - Prefer high-signal features, not boilerplate preprocessing.
    - Use the dataset summary, prior accepted steps, and prior reviewer feedback if provided.
    - Do not import modules. Assume `pd` and `np` already exist.
    - Do not read/write files, call external APIs, or use shell commands.
    - Each step should gracefully skip when required columns are absent.
    """
).strip()


REVIEWER_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are the reviewer in a two-agent EDA system.
    Critique the planner proposal, remove weak or risky ideas, and return a tighter revised plan.
    Respond ONLY with valid JSON.

    {BASE_LLM_RESPONSE_SCHEMA.strip()}

    Reviewer rules:
    - Prefer fewer, stronger steps over many noisy ones.
    - Remove schema-fragile, target-leaking, or redundant transformations.
    - If the planner is already strong, keep the best steps and improve the rationale.
    - Do not import modules. Assume `pd` and `np` already exist.
    - Do not read/write files, call external APIs, or use shell commands.
    - Each step must be safe for unseen tabular schemas.
    """
).strip()


def _call_openai_compatible(
    payload_text: str,
    system_prompt: str,
    model_env_names: tuple[str, ...] = ("EDA_MODEL_NAME", "MODEL_NAME"),
) -> dict[str, Any]:
    api_base_url = get_runtime_secret("API_BASE_URL")
    model_name = get_runtime_secret(*model_env_names)
    api_key = get_runtime_secret("OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN")
    if not api_base_url or not model_name or not api_key:
        raise RuntimeError(
            "OpenAI-compatible EDA call requires API_BASE_URL, MODEL_NAME/EDA_MODEL_NAME, and an API key."
        )

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


def _call_anthropic(
    payload_text: str,
    system_prompt: str,
    model_env_name: str = "ANTHROPIC_MODEL",
) -> dict[str, Any]:
    import urllib.request

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    body = json.dumps(
        {
            "model": os.getenv(model_env_name, os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")),
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


def _call_llm_custom(
    payload_text: str,
    system_prompt: str,
    model_env_names: tuple[str, ...] = ("EDA_MODEL_NAME", "MODEL_NAME"),
    anthropic_model_env: str = "ANTHROPIC_MODEL",
) -> tuple[dict[str, Any], str]:
    provider = os.getenv("EDA_PROVIDER", "").strip().lower()
    if provider == "anthropic":
        return _call_anthropic(payload_text, system_prompt, model_env_name=anthropic_model_env), "anthropic"
    if provider == "openai":
        return _call_openai_compatible(payload_text, system_prompt, model_env_names=model_env_names), "openai-compatible"

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _call_anthropic(payload_text, system_prompt, model_env_name=anthropic_model_env), "anthropic"
        except Exception:
            pass
    return _call_openai_compatible(payload_text, system_prompt, model_env_names=model_env_names), "openai-compatible"


def _call_llm(payload_text: str) -> tuple[dict[str, Any], str]:
    return _call_llm_custom(payload_text, SINGLE_PASS_SYSTEM_PROMPT)


def _sanitize_feature_code(code: str) -> str:
    cleaned = code.strip()
    cleaned = cleaned.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    sanitized_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped in {"import pandas as pd", "import numpy as np"}:
            continue
        if stripped.startswith("from pandas import") or stripped.startswith("from numpy import"):
            continue
        if stripped.startswith("return "):
            sanitized_lines.append(line.replace("return ", "", 1))
            continue
        sanitized_lines.append(line)

    return "\n".join(sanitized_lines).strip()


def _normalize_text_list(values: Any, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if text and key not in seen:
            normalized.append(text)
            seen.add(key)
        if len(normalized) >= limit:
            break
    return normalized


def _is_safe_ast(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    banned_nodes = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.TryStar)
    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            return False
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in {"open", "eval", "exec", "compile", "input"}:
                return False
    return True


def _apply_feature_step(df: pd.DataFrame, step: FeatureEngineeringStep) -> pd.DataFrame:
    local_ns: dict[str, Any] = {"df": df.copy(), "pd": pd, "np": np}
    exec(step.code, local_ns, local_ns)  # noqa: S102
    transformed = local_ns["df"]
    if not isinstance(transformed, pd.DataFrame):
        raise TypeError("Feature step did not produce a DataFrame.")
    return transformed


def _build_planner_payload(
    payload_text: str,
    accepted_steps: list[FeatureEngineeringStep],
    prior_feedback: list[str],
    round_index: int,
) -> str:
    accepted_descriptions = [step.description for step in accepted_steps] or ["None yet."]
    feedback_lines = prior_feedback[-5:] or ["None yet."]
    return textwrap.dedent(
        f"""
        You are on round {round_index} of a bounded EDA planning loop.

        {payload_text}

        Accepted feature steps so far:
        {json.dumps(accepted_descriptions, indent=2)}

        Reviewer feedback from previous rounds:
        {json.dumps(feedback_lines, indent=2)}
        """
    ).strip()


def _build_reviewer_payload(
    payload_text: str,
    planner_result: dict[str, Any],
    accepted_steps: list[FeatureEngineeringStep],
    round_index: int,
) -> str:
    accepted_descriptions = [step.description for step in accepted_steps] or ["None yet."]
    return textwrap.dedent(
        f"""
        You are reviewing round {round_index} of a bounded EDA planning loop.

        {payload_text}

        Accepted feature steps so far:
        {json.dumps(accepted_descriptions, indent=2)}

        Planner proposal to critique and tighten:
        {json.dumps(planner_result, indent=2)}
        """
    ).strip()


def _validate_feature_step(
    df: pd.DataFrame,
    target: str | None,
    step: FeatureEngineeringStep,
) -> tuple[bool, str, FeatureEngineeringStep, pd.DataFrame | None]:
    code = _sanitize_feature_code(step.code)
    if not step.description.strip() or not step.rationale.strip() or not code:
        return False, "Missing description, rationale, or code.", step, None
    lowered = code.lower()
    if any(token in lowered for token in BLOCKED_STEP_TOKENS):
        return False, "Blocked unsafe token in code.", step, None
    if not _is_safe_ast(code):
        return False, "Code failed safety AST validation.", step, None

    candidate = df.copy()
    original_rows = len(candidate)
    original_columns = list(candidate.columns)
    local_ns: dict[str, Any] = {"df": candidate, "pd": pd, "np": np}
    try:
        exec(code, local_ns, local_ns)  # noqa: S102
    except Exception as exc:
        return False, f"Execution failed: {exc}", step, None

    transformed = local_ns.get("df")
    if not isinstance(transformed, pd.DataFrame):
        return False, "Step did not return a DataFrame.", step, None
    if len(transformed) != original_rows:
        return False, "Row count changed unexpectedly.", step, None
    if transformed.columns.duplicated().any():
        return False, "Produced duplicate column names.", step, None
    if target and target in original_columns and target not in transformed.columns:
        return False, "Dropped the target column.", step, None
    if transformed.shape[1] < max(1, len(original_columns) // 2):
        return False, "Dropped too many columns.", step, None
    return (
        True,
        "accepted",
        FeatureEngineeringStep(
            description=step.description,
            rationale=step.rationale,
            code=code,
        ),
        transformed,
    )


def _validated_llm_steps(
    df: pd.DataFrame,
    target: str | None,
    raw_steps: list[dict[str, Any]],
    existing_descriptions: set[str] | None = None,
) -> tuple[list[FeatureEngineeringStep], int, list[str], pd.DataFrame]:
    accepted: list[FeatureEngineeringStep] = []
    rejected = 0
    rejection_reasons: list[str] = []
    seen_descriptions = {description.strip().lower() for description in (existing_descriptions or set()) if description.strip()}
    working_df = df.copy()

    for raw_step in raw_steps:
        step = FeatureEngineeringStep(
            description=str(raw_step.get("description", "")).strip(),
            rationale=str(raw_step.get("rationale", "")).strip(),
            code=str(raw_step.get("code", "df")).strip(),
        )
        description_key = step.description.strip().lower()
        if description_key and description_key in seen_descriptions:
            continue

        is_valid, reason, validated_step, transformed = _validate_feature_step(working_df, target, step)
        if is_valid:
            accepted.append(validated_step)
            working_df = transformed if transformed is not None else working_df
            if description_key:
                seen_descriptions.add(description_key)
        else:
            rejected += 1
            label = step.description or f"step_{rejected}"
            rejection_reasons.append(f"{label}: {reason}")
    return accepted, rejected, rejection_reasons[:8], working_df


def _merge_feature_steps(
    heuristic_steps: list[FeatureEngineeringStep],
    llm_steps: list[FeatureEngineeringStep],
) -> list[FeatureEngineeringStep]:
    merged: list[FeatureEngineeringStep] = []
    seen_descriptions: set[str] = set()
    for step in [*heuristic_steps, *llm_steps]:
        key = step.description.strip().lower()
        if key and key not in seen_descriptions:
            merged.append(step)
            seen_descriptions.add(key)
    return merged[:8]


def _run_planner_reviewer_loop(
    df: pd.DataFrame,
    target: str | None,
    payload_text: str,
    heuristic_steps: list[FeatureEngineeringStep],
    base_recommendations: list[str],
    max_rounds: int,
) -> dict[str, Any]:
    accepted_steps: list[FeatureEngineeringStep] = []
    working_df = df.copy()
    rejection_reasons: list[str] = []
    round_records: list[LLMRoundRecord] = []
    all_recommendations = list(base_recommendations)
    all_summaries: list[str] = []
    candidate_step_total = 0
    llm_provider = "none"
    prior_feedback: list[str] = []
    inferred_task_type = _infer_task_type(df, target)

    for round_index in range(1, max_rounds + 1):
        planner_payload = _build_planner_payload(payload_text, accepted_steps, prior_feedback, round_index)
        planner_result, planner_provider = _call_llm_custom(
            planner_payload,
            PLANNER_SYSTEM_PROMPT,
            model_env_names=("EDA_PLANNER_MODEL_NAME", "EDA_MODEL_NAME", "MODEL_NAME"),
            anthropic_model_env="ANTHROPIC_PLANNER_MODEL",
        )
        if llm_provider == "none":
            llm_provider = planner_provider

        planner_steps = list(planner_result.get("feature_engineering_steps", []))
        planner_summary = str(planner_result.get("summary", "")).strip()
        planner_recommendations = _normalize_text_list(planner_result.get("recommendations", []))
        inferred_task_type = str(planner_result.get("task_type", inferred_task_type))

        reviewer_payload = _build_reviewer_payload(payload_text, planner_result, accepted_steps, round_index)
        try:
            reviewer_result, reviewer_provider = _call_llm_custom(
                reviewer_payload,
                REVIEWER_SYSTEM_PROMPT,
                model_env_names=("EDA_REVIEWER_MODEL_NAME", "EDA_REVIEW_MODEL_NAME", "EDA_MODEL_NAME", "MODEL_NAME"),
                anthropic_model_env="ANTHROPIC_REVIEWER_MODEL",
            )
            if llm_provider == "none":
                llm_provider = reviewer_provider
        except Exception as exc:
            reviewer_result = {
                "summary": f"Reviewer unavailable in round {round_index}: {exc}. Validating planner output directly.",
                "recommendations": ["Proceed with the strongest planner steps that pass validation."],
                "feature_engineering_steps": planner_steps,
            }

        reviewer_steps = list(reviewer_result.get("feature_engineering_steps", [])) or planner_steps
        reviewer_summary = str(reviewer_result.get("summary", "")).strip()
        reviewer_recommendations = _normalize_text_list(reviewer_result.get("recommendations", []))
        inferred_task_type = str(reviewer_result.get("task_type", inferred_task_type))
        candidate_step_total += len(reviewer_steps)

        validated_steps, rejected_count, round_rejections, working_df = _validated_llm_steps(
            working_df,
            target,
            reviewer_steps,
            existing_descriptions={step.description for step in accepted_steps},
        )
        accepted_steps.extend(validated_steps)
        rejection_reasons.extend(round_rejections)

        all_recommendations.extend(planner_recommendations)
        all_recommendations.extend(reviewer_recommendations)
        if planner_summary:
            all_summaries.append(planner_summary)
        if reviewer_summary:
            all_summaries.append(reviewer_summary)
        prior_feedback = reviewer_recommendations or ([reviewer_summary] if reviewer_summary else prior_feedback)

        round_records.append(
            LLMRoundRecord(
                round_index=round_index,
                planner_summary=planner_summary or "Planner returned no summary.",
                planner_recommendations=planner_recommendations,
                planner_step_count=len(planner_steps),
                reviewer_summary=reviewer_summary or "Reviewer returned no summary.",
                reviewer_recommendations=reviewer_recommendations,
                reviewer_step_count=len(reviewer_steps),
                accepted_step_count=len(validated_steps),
                rejected_step_count=rejected_count,
            )
        )

        if not validated_steps and round_index >= 2:
            break

    final_recommendations = _normalize_text_list(all_recommendations, limit=12)
    summary_prefix = (
        "Planner-reviewer EDA loop completed with statistical profiling as the source of truth."
    )
    if all_summaries:
        summary_prefix = f"{all_summaries[-1]} Statistical profiling remained the source of truth throughout the planner-reviewer loop."

    return {
        "llm_provider": llm_provider,
        "task_type": inferred_task_type,
        "agent_summary": summary_prefix,
        "agent_recommendations": final_recommendations,
        "validated_steps": accepted_steps,
        "validated_step_count": len(accepted_steps),
        "candidate_step_total": candidate_step_total,
        "rejected_step_count": len(rejection_reasons),
        "rejection_reasons": rejection_reasons[:12],
        "round_records": round_records,
        "rounds_run": len(round_records),
    }


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
        f"- LLM strategy: `{report.llm_strategy}`",
        f"- LLM rounds run: `{report.llm_rounds_run}`",
        f"- LLM candidate steps: `{report.llm_candidate_steps}`",
        f"- Validated LLM steps: `{report.validated_llm_steps}`",
        f"- Rejected LLM steps: `{report.rejected_llm_steps}`",
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
    if report.llm_rejection_reasons:
        markdown.extend(
            [
                "## Rejected LLM Steps",
                "",
                *[f"- {reason}" for reason in report.llm_rejection_reasons],
                "",
            ]
        )
    if report.llm_round_records:
        markdown.extend(["## Planner / Reviewer Dialogue", ""])
        for record in report.llm_round_records:
            markdown.extend(
                [
                    f"### Round {record.round_index}",
                    "",
                    f"- Planner summary: {record.planner_summary}",
                    f"- Planner recommendations: {', '.join(record.planner_recommendations) if record.planner_recommendations else 'None'}",
                    f"- Planner candidate steps: {record.planner_step_count}",
                    f"- Reviewer summary: {record.reviewer_summary}",
                    f"- Reviewer recommendations: {', '.join(record.reviewer_recommendations) if record.reviewer_recommendations else 'None'}",
                    f"- Reviewer candidate steps: {record.reviewer_step_count}",
                    f"- Accepted after validation: {record.accepted_step_count}",
                    f"- Rejected after validation: {record.rejected_step_count}",
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

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        use_llm: bool = True,
        llm_strategy: str = "single_pass",
        max_llm_rounds: int = 2,
    ) -> None:
        self.df = df.copy()
        self.target = target_column
        self.use_llm = use_llm
        self.llm_strategy = llm_strategy.strip().lower() if llm_strategy else "single_pass"
        self.max_llm_rounds = max(1, min(int(max_llm_rounds), 3))

    def run(self) -> EDAReport:
        print("  [EDAAgent] Profiling columns…")
        profiles = _profile_columns(self.df, self.target)
        print("  [EDAAgent] Computing correlations…")
        correlation_insights = _correlation_insights(self.df, self.target)
        print("  [EDAAgent] Detecting quality issues…")
        quality_issues = _detect_quality_issues(self.df, self.target)

        feature_steps = _heuristic_feature_steps(self.df, self.target, profiles)
        agent_summary = (
            "Deterministic statistical EDA complete. Using schema-grounded heuristic recommendations."
        )
        agent_recommendations = [
            "Start with missing-value repair and schema review before feature engineering.",
            "Review high-cardinality columns in batches rather than attempting the full schema at once.",
        ]
        task_type = _infer_task_type(self.df, self.target)
        llm_provider = "none"
        llm_strategy = "none"
        llm_rounds_run = 0
        llm_candidate_steps = 0
        validated_llm_steps = 0
        rejected_llm_steps = 0
        llm_rejection_reasons: list[str] = []
        llm_round_records: list[LLMRoundRecord] = []

        if self.use_llm:
            print("  [EDAAgent] Calling LLM for schema-grounded suggestions…")
            payload = _build_prompt_payload(self.df, self.target, profiles, correlation_insights, quality_issues)
            try:
                if self.llm_strategy == "planner_reviewer":
                    print(
                        f"  [EDAAgent] Running planner-reviewer loop for up to {self.max_llm_rounds} rounds…"
                    )
                    loop_result = _run_planner_reviewer_loop(
                        self.df,
                        self.target,
                        payload,
                        heuristic_steps=feature_steps,
                        base_recommendations=agent_recommendations,
                        max_rounds=self.max_llm_rounds,
                    )
                    llm_strategy = "planner_reviewer"
                    llm_provider = str(loop_result["llm_provider"])
                    task_type = str(loop_result.get("task_type", task_type))
                    llm_rounds_run = int(loop_result["rounds_run"])
                    llm_candidate_steps = int(loop_result["candidate_step_total"])
                    validated_llm_steps = int(loop_result["validated_step_count"])
                    rejected_llm_steps = int(loop_result["rejected_step_count"])
                    llm_rejection_reasons = list(loop_result["rejection_reasons"])
                    llm_round_records = list(loop_result["round_records"])
                    feature_steps = _merge_feature_steps(feature_steps, list(loop_result["validated_steps"]))
                    agent_summary = str(loop_result["agent_summary"])
                    agent_recommendations = list(loop_result["agent_recommendations"])
                    print(
                        f"  [EDAAgent] Planner-reviewer loop accepted {validated_llm_steps} validated steps and rejected {rejected_llm_steps}."
                    )
                else:
                    result, llm_provider = _call_llm(payload)
                    llm_strategy = "single_pass"
                    llm_rounds_run = 1
                    task_type = result.get("task_type", task_type)
                    llm_summary = str(result.get("summary", "")).strip()
                    if llm_summary:
                        agent_summary = (
                            f"{llm_summary} Statistical profiling remained the source of truth, "
                            "and only validated LLM transformations were retained."
                        )
                    llm_recommendations = _normalize_text_list(result.get("recommendations", []))
                    if llm_recommendations:
                        agent_recommendations = agent_recommendations + llm_recommendations
                    raw_steps = list(result.get("feature_engineering_steps", []))
                    llm_candidate_steps = len(raw_steps)
                    validated_steps, rejected_llm_steps, llm_rejection_reasons, _ = _validated_llm_steps(
                        self.df,
                        self.target,
                        raw_steps,
                    )
                    validated_llm_steps = len(validated_steps)
                    feature_steps = _merge_feature_steps(feature_steps, validated_steps)
                    print(
                        f"  [EDAAgent] Accepted {validated_llm_steps} validated LLM steps and rejected {rejected_llm_steps}."
                    )
            except Exception as exc:
                agent_summary = (
                    f"LLM analysis unavailable ({exc}). Falling back to schema-grounded heuristic EDA recommendations."
                )
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
            llm_strategy=llm_strategy,
            llm_rounds_run=llm_rounds_run,
            llm_candidate_steps=llm_candidate_steps,
            validated_llm_steps=validated_llm_steps,
            rejected_llm_steps=rejected_llm_steps,
            llm_rejection_reasons=llm_rejection_reasons,
            llm_round_records=llm_round_records,
        )


def main() -> None:
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    target = sys.argv[2] if len(sys.argv) > 2 else None
    use_llm = len(sys.argv) <= 3 or sys.argv[3].lower() != "false"
    llm_strategy = sys.argv[4] if len(sys.argv) > 4 else "single_pass"
    max_llm_rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 2

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

    report = EDAAgent(
        df,
        target_column=target,
        use_llm=use_llm,
        llm_strategy=llm_strategy,
        max_llm_rounds=max_llm_rounds,
    ).run()
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
