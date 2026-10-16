"""Leakage-safe dataset preparation for train-ready CSV outputs.

Raw rows are split before anything is learned. Every statistic and decision
(fill values, clip bounds, category lists, frequency maps, which columns become
which features) is fitted on training rows only, stored in a PreprocessorState,
and replayed unchanged on validation, cross-validation, and inference rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .eda_agent import EDAAgent, EDAReport, FeatureEngineeringStep, apply_feature_steps, write_eda_artifacts
from .reporting import build_dataset_profile, build_work_queue, write_profile_bundle, write_report_bundle


LOW_CARDINALITY_THRESHOLD = 12
HIGH_UNIQUE_RATIO = 0.95
DATE_PARSE_THRESHOLD = 0.8
NUMERIC_PARSE_THRESHOLD = 0.9

TASK_TYPES = ("auto", "classification", "regression")
MODES = ("general", "credit_scoring")
SPLIT_STRATEGIES = ("auto", "stratified", "random", "group", "time")
SOURCE_ROW_COLUMN = "source_row"
CREDIT_TARGET_DEFINITION = "1 = borrower defaults within 24 months, 0 = no default within 24 months."
# Name fragments that usually mean the column is only known after the loan
# outcome. They are flagged for review, not dropped: only the user can confirm
# what a column means and when it becomes available.
POST_OUTCOME_TOKENS = (
    "recover",
    "collection",
    "charge_off",
    "chargeoff",
    "charged_off",
    "write_off",
    "writeoff",
    "settle",
    "default_date",
    "loan_status",
    "paid_off",
    "last_payment",
    "outcome",
)
TRUTHY = {"true", "yes", "y", "1"}
FALSY = {"false", "no", "n", "0"}


@dataclass
class PrepConfig:
    target_column: str | None = None
    task_type: str = "auto"
    mode: str = "general"
    split_strategy: str = "auto"
    group_column: str | None = None
    time_column: str | None = None
    id_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    ratio_features: list[str] = field(default_factory=list)
    validation_fraction: float = 0.2
    random_seed: int = 42
    use_eda_agent: bool = False
    eda_use_llm: bool = False
    eda_llm_strategy: str = "single_pass"
    eda_llm_rounds: int = 2

    @property
    def passthrough_columns(self) -> list[str]:
        """Columns carried next to the features but never used as features."""
        columns = [SOURCE_ROW_COLUMN, *self.id_columns, self.group_column, self.time_column]
        return list(dict.fromkeys(column for column in columns if column))

    @property
    def ratio_pairs(self) -> list[tuple[str, str]]:
        return [tuple(item.split(":", 1)) for item in self.ratio_features]  # type: ignore[misc]


@dataclass
class PreprocessorState:
    """Everything learned from training rows, in a JSON-safe form."""

    input_columns: list[str]
    passthrough_columns: list[str] = field(default_factory=list)
    ratio_features: list[list[str]] = field(default_factory=list)
    eda_steps: list[dict[str, str]] = field(default_factory=list)
    boolean_columns: list[str] = field(default_factory=list)
    numeric_coerced_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_length_columns: list[str] = field(default_factory=list)
    missing_indicator_columns: list[str] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)
    all_missing_columns: list[str] = field(default_factory=list)
    clip_bounds: dict[str, list[float]] = field(default_factory=dict)
    numeric_fill: dict[str, float] = field(default_factory=dict)
    one_hot: dict[str, dict[str, Any]] = field(default_factory=dict)
    frequency: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    task_type: str | None = None
    target_mapping: dict[str, int] | None = None
    fitted_rows: int = 0

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(json.dumps(asdict(self), indent=2))
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessorState":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class SplitFrame:
    passthrough: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series | None

    def combined(self) -> pd.DataFrame:
        parts = [self.passthrough.reset_index(drop=True), self.features.reset_index(drop=True)]
        if self.target is not None:
            parts.append(self.target.reset_index(drop=True))
        return pd.concat(parts, axis=1)


@dataclass
class SplitResult:
    state: PreprocessorState
    steps: list[str]
    eda_report: EDAReport | None
    eda_steps: list[FeatureEngineeringStep]
    train: SplitFrame
    valid: SplitFrame | None
    duplicate_rows_removed: int
    unseen_categories_in_valid: dict[str, int]


@dataclass
class DatasetPreparationArtifacts:
    summary: dict[str, Any]
    prepared_full_path: Path
    prepared_train_path: Path | None
    prepared_valid_path: Path | None
    manifest_path: Path
    config: PrepConfig | None = None

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


def _looks_like_identifier(series: pd.Series, column_name: str) -> bool:
    lowered = column_name.lower()
    unique_ratio = float(series.nunique(dropna=False) / max(1, len(series)))
    if unique_ratio < HIGH_UNIQUE_RATIO:
        return False
    return any(token in lowered for token in ("id", "uuid", "guid", "ticket", "record", "name"))


def _is_plain_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


# ---------------------------------------------------------------------------
# Configuration and splitting
# ---------------------------------------------------------------------------


def load_raw_frame(csv_path: str | Path) -> tuple[Path, pd.DataFrame]:
    source_path = Path(csv_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")
    raw_df = pd.read_csv(source_path)
    raw_df.columns = [str(column).strip() for column in raw_df.columns]
    if SOURCE_ROW_COLUMN in raw_df.columns:
        raise ValueError(f"Input already has a '{SOURCE_ROW_COLUMN}' column; it is reserved for row tracking.")
    raw_df.insert(0, SOURCE_ROW_COLUMN, np.arange(len(raw_df)))
    return source_path, raw_df


def validate_config(config: PrepConfig, df: pd.DataFrame) -> None:
    if config.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{config.mode}'.")
    if config.task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}, got '{config.task_type}'.")
    if config.split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy must be one of {SPLIT_STRATEGIES}, got '{config.split_strategy}'.")
    if not 0.0 < float(config.validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    for item in config.ratio_features:
        if item.count(":") != 1 or not all(part.strip() for part in item.split(":")):
            raise ValueError(f"Ratio feature '{item}' must look like 'numerator_column:denominator_column'.")

    referenced = {
        "target_column": [config.target_column] if config.target_column else [],
        "id_columns": config.id_columns,
        "group_column": [config.group_column] if config.group_column else [],
        "time_column": [config.time_column] if config.time_column else [],
        "exclude_columns": config.exclude_columns,
        "ratio_features": [part for pair in config.ratio_pairs for part in pair],
    }
    for setting, columns in referenced.items():
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"{setting} references columns not in the dataset: {', '.join(missing)}.")

    target = config.target_column
    if target and target in {*config.id_columns, config.group_column, config.time_column, *config.exclude_columns}:
        raise ValueError(f"Target column '{target}' cannot also be an id, group, time, or excluded column.")
    if target and any(target in pair for pair in config.ratio_pairs):
        raise ValueError(f"Target column '{target}' cannot be used in a ratio feature.")

    if config.mode == "credit_scoring":
        if not target:
            raise ValueError("credit_scoring mode requires a target column.")
        if config.task_type == "regression":
            raise ValueError("credit_scoring mode is binary classification; task_type cannot be regression.")
    if config.split_strategy == "group" and not config.group_column:
        raise ValueError("split_strategy 'group' requires group_column.")
    if config.split_strategy == "time" and not config.time_column:
        raise ValueError("split_strategy 'time' requires time_column.")
    if config.split_strategy == "auto" and config.group_column and config.time_column:
        raise ValueError("Both group_column and time_column are set; choose split_strategy 'group' or 'time'.")


def resolve_task_type(target: pd.Series, config: PrepConfig) -> str:
    if target.isna().any():
        raise ValueError(
            f"Target column '{config.target_column}' has {int(target.isna().sum())} missing labels; "
            "fix or remove them before preparation."
        )
    if config.mode == "credit_scoring":
        numeric = pd.to_numeric(target, errors="coerce")
        labels = set(numeric.dropna().unique().tolist())
        if numeric.isna().any() or labels != {0, 1}:
            found = sorted(map(str, target.unique().tolist()))[:10]
            raise ValueError(
                "credit_scoring mode needs a binary 0/1 target "
                f"({CREDIT_TARGET_DEFINITION}). Found labels: {', '.join(found)}."
            )
        return "classification"
    if config.task_type != "auto":
        return config.task_type
    if not is_numeric_dtype(target) or target.nunique() <= 20:
        return "classification"
    return "regression"


def resolve_split_strategy(config: PrepConfig, task_type: str) -> str:
    strategy = config.split_strategy
    if strategy == "auto":
        if config.group_column:
            return "group"
        if config.time_column:
            return "time"
        return "stratified" if task_type == "classification" else "random"
    if strategy == "stratified" and task_type != "classification":
        raise ValueError("Stratified splitting needs a classification target.")
    return strategy


def parse_time_column(series: pd.Series, column: str) -> pd.Series:
    parsed = series if is_numeric_dtype(series) else pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"Time column '{column}' has {int(parsed.isna().sum())} missing or unparseable values; "
            "chronological splits need a timestamp on every row."
        )
    return parsed


def split_rows(df: pd.DataFrame, config: PrepConfig, task_type: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return sorted (train_positions, valid_positions, strategy) over df rows."""
    strategy = resolve_split_strategy(config, task_type)
    n_rows = len(df)
    positions = np.arange(n_rows)
    n_valid = max(1, int(round(n_rows * config.validation_fraction)))
    target = df[config.target_column] if config.target_column else None

    if strategy == "stratified":
        counts = target.astype(str).value_counts()
        if counts.min() < 2:
            raise ValueError(
                f"Stratified split needs at least 2 rows per class; class '{counts.idxmin()}' has {counts.min()}."
            )
        n_valid = max(n_valid, len(counts))
        if n_rows - n_valid < len(counts):
            raise ValueError("Dataset is too small for a stratified split that keeps every class in both splits.")
        train_pos, valid_pos = train_test_split(
            positions, test_size=n_valid, stratify=target.astype(str), random_state=config.random_seed
        )
    elif strategy == "random":
        train_pos, valid_pos = train_test_split(positions, test_size=n_valid, random_state=config.random_seed)
    elif strategy == "group":
        groups = df[config.group_column]
        if groups.isna().any():
            raise ValueError(f"Group column '{config.group_column}' has missing values; every row needs a group.")
        if groups.nunique() < 2:
            raise ValueError(f"Group column '{config.group_column}' needs at least 2 distinct groups to split.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.validation_fraction, random_state=config.random_seed)
        train_pos, valid_pos = next(splitter.split(positions, groups=groups.astype(str)))
    else:
        times = parse_time_column(df[config.time_column], config.time_column)
        counts_by_time = times.value_counts().sort_index(ascending=False)
        cutoff = counts_by_time.index[-1]
        cumulative = 0
        for timestamp, count in counts_by_time.items():
            cumulative += int(count)
            cutoff = timestamp
            if cumulative >= n_valid:
                break
        is_valid = (times >= cutoff).to_numpy()
        train_pos, valid_pos = positions[~is_valid], positions[is_valid]

    if len(train_pos) == 0 or len(valid_pos) == 0:
        raise ValueError(f"{strategy} split produced an empty train or validation set; adjust validation_fraction.")
    return np.sort(train_pos), np.sort(valid_pos), strategy


def find_post_outcome_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if any(token in _slugify(column) for token in POST_OUTCOME_TOKENS)]


# ---------------------------------------------------------------------------
# Transform stages: each replays a decision stored in PreprocessorState
# ---------------------------------------------------------------------------


def _ratio_name(numerator: str, denominator: str) -> str:
    return f"{_slugify(numerator)}__per__{_slugify(denominator)}"


def _apply_ratios(df: pd.DataFrame, pairs: list[list[str]] | list[tuple[str, str]]) -> pd.DataFrame:
    for numerator, denominator in pairs:
        top = pd.to_numeric(df[numerator], errors="coerce")
        bottom = pd.to_numeric(df[denominator], errors="coerce").replace(0, np.nan)
        df[_ratio_name(numerator, denominator)] = top / bottom
    return df


def _to_bool_float(series: pd.Series) -> pd.Series:
    lowered = series.astype("string").str.strip().str.lower()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result[lowered.isin(TRUTHY).fillna(False).to_numpy(dtype=bool)] = 1.0
    result[lowered.isin(FALSY).fillna(False).to_numpy(dtype=bool)] = 0.0
    return result


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _apply_datetimes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        parsed = _parse_dates(df[column])
        base = _slugify(column)
        df[f"{base}__year"] = parsed.dt.year.astype(float)
        df[f"{base}__month"] = parsed.dt.month.astype(float)
        df[f"{base}__day"] = parsed.dt.day.astype(float)
        df[f"{base}__weekday"] = parsed.dt.weekday.astype(float)
    return df.drop(columns=columns)


def _categorical_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").astype(object).where(series.notna(), None)


def _apply_stages(
    raw_features: pd.DataFrame,
    state: PreprocessorState,
    upto: str | None = None,
    unseen: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Replay fitted stages in order. `upto` stops before the named stage (used while fitting)."""
    df = raw_features[state.input_columns].copy()
    df = _apply_ratios(df, state.ratio_features)
    if state.eda_steps:
        df = apply_feature_steps(df, [FeatureEngineeringStep(**step) for step in state.eda_steps], strict=True)
    if upto == "booleans":
        return df
    for column in state.boolean_columns:
        df[column] = _to_bool_float(df[column])
    if upto == "numeric":
        return df
    for column in state.numeric_coerced_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if upto == "datetimes":
        return df
    df = _apply_datetimes(df, state.datetime_columns)
    if upto == "text_lengths":
        return df
    for column in state.text_length_columns:
        df[f"{_slugify(column)}__len"] = df[column].fillna("").astype(str).str.len()
    if upto == "missing_indicators":
        return df
    for column in state.missing_indicator_columns:
        df[f"{_slugify(column)}__was_missing"] = df[column].isna().astype(int)
    if upto == "drops":
        return df
    df = df.drop(columns=[*state.identifier_columns, *state.all_missing_columns])
    if upto == "clip":
        return df
    for column, (lower, upper) in state.clip_bounds.items():
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(lower=lower, upper=upper)
    if upto == "fill":
        return df
    for column, value in state.numeric_fill.items():
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(value)
    if upto == "encode":
        return df

    for column, spec in state.one_hot.items():
        values = _categorical_strings(df[column])
        known = set(spec["categories"])
        if unseen is not None:
            unseen[column] = unseen.get(column, 0) + int((values.notna() & ~values.isin(known)).sum())
        values = values.fillna(spec["fill"])
        prefix = _slugify(column)
        for category in spec["categories"]:
            df[f"{prefix}_{category}"] = (values == category).astype(int)
        df = df.drop(columns=[column])
    for column, spec in state.frequency.items():
        values = _categorical_strings(df[column])
        if unseen is not None:
            unseen[column] = unseen.get(column, 0) + int((values.notna() & ~values.isin(set(spec["map"]))).sum())
        values = values.fillna(spec["fill"])
        df[f"{_slugify(column)}__freq"] = values.map(spec["map"]).astype(float).fillna(0.0)
        df = df.drop(columns=[column])
    return df


def transform_features(raw_features: pd.DataFrame, state: PreprocessorState) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply fitted preprocessing unchanged. Returns features in fitted order and unseen-category counts."""
    missing = [column for column in state.input_columns if column not in raw_features.columns]
    if missing:
        raise ValueError(f"Rows are missing columns required by the fitted preprocessing: {', '.join(missing)}.")
    unseen: dict[str, int] = {}
    transformed = _apply_stages(raw_features, state, unseen=unseen)
    if set(transformed.columns) != set(state.feature_columns):
        extra = sorted(set(transformed.columns) - set(state.feature_columns))
        lost = sorted(set(state.feature_columns) - set(transformed.columns))
        raise ValueError(f"Transformed features do not match fitted features (extra={extra}, missing={lost}).")
    return transformed[state.feature_columns].reset_index(drop=True), {k: v for k, v in unseen.items() if v}


# ---------------------------------------------------------------------------
# Fitting: learn every decision from training rows only
# ---------------------------------------------------------------------------


def fit_preprocessor(
    train_features: pd.DataFrame,
    ratio_pairs: list[tuple[str, str]],
    eda_steps: list[FeatureEngineeringStep],
    steps: list[str],
) -> PreprocessorState:
    state = PreprocessorState(
        input_columns=list(train_features.columns),
        ratio_features=[list(pair) for pair in ratio_pairs],
        eda_steps=[{"description": s.description, "rationale": s.rationale, "code": s.code} for s in eda_steps],
        fitted_rows=int(len(train_features)),
    )
    if ratio_pairs:
        steps.append(f"Added {len(ratio_pairs)} requested ratio features (row-wise, zero denominators become missing).")

    df = _apply_stages(train_features, state, upto="booleans")
    for column in df.columns:
        series = df[column]
        if is_bool_dtype(series):
            state.boolean_columns.append(column)
            continue
        if is_numeric_dtype(series):
            continue
        lowered = set(series.dropna().astype(str).str.strip().str.lower().unique())
        if lowered and lowered.issubset(TRUTHY | FALSY):
            state.boolean_columns.append(column)
    if state.boolean_columns:
        steps.append(f"Mapped boolean-like columns to 1/0, keeping missing values missing: {', '.join(state.boolean_columns)}.")

    df = _apply_stages(train_features, state, upto="numeric")
    for column in df.columns:
        series = df[column]
        if is_numeric_dtype(series) or series.notna().sum() == 0:
            continue
        non_null = series.dropna()
        if pd.to_numeric(non_null, errors="coerce").notna().mean() >= NUMERIC_PARSE_THRESHOLD:
            state.numeric_coerced_columns.append(column)
            steps.append(f"Converted column '{column}' to numeric values.")

    df = _apply_stages(train_features, state, upto="datetimes")
    for column in df.columns:
        series = df[column]
        if is_numeric_dtype(series):
            continue
        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            continue
        pattern = r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]{3,9}\s+\d{1,2})"
        if sample.str.contains(pattern, regex=True).mean() < 0.5:
            continue
        parsed = _parse_dates(series.dropna())
        if parsed.notna().mean() >= DATE_PARSE_THRESHOLD and parsed.nunique(dropna=True) > 1:
            state.datetime_columns.append(column)
            steps.append(f"Expanded datetime column '{column}' into calendar features.")

    df = _apply_stages(train_features, state, upto="text_lengths")
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            continue
        non_null = df[column].dropna().astype(str)
        if non_null.empty:
            continue
        if non_null.str.len().mean() >= 10 and non_null.nunique() / max(1, len(non_null)) > 0.5:
            state.text_length_columns.append(column)
    if state.text_length_columns:
        steps.append(f"Added text-length features for: {', '.join(state.text_length_columns)}.")

    df = _apply_stages(train_features, state, upto="missing_indicators")
    state.missing_indicator_columns = [
        column for column in df.columns if df[column].isna().any() and df[column].notna().any()
    ]
    if state.missing_indicator_columns:
        steps.append(f"Added missingness indicators for {len(state.missing_indicator_columns)} columns.")

    df = _apply_stages(train_features, state, upto="drops")
    state.all_missing_columns = [column for column in df.columns if df[column].isna().all()]
    state.identifier_columns = [
        column
        for column in df.columns
        if column not in state.all_missing_columns and _looks_like_identifier(df[column], column)
    ]
    if state.all_missing_columns:
        steps.append(f"Dropped columns with no training values: {', '.join(state.all_missing_columns)}.")
    if state.identifier_columns:
        steps.append(f"Dropped identifier-like columns: {', '.join(state.identifier_columns)}.")

    df = _apply_stages(train_features, state, upto="clip")
    for column in df.columns:
        if not _is_plain_numeric(df[column]) or df[column].nunique(dropna=True) < 4:
            continue
        lower, upper = df[column].quantile(0.01), df[column].quantile(0.99)
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            state.clip_bounds[column] = [float(lower), float(upper)]
    if state.clip_bounds:
        steps.append(f"Clipped extreme numeric outliers in {len(state.clip_bounds)} columns (training 1st/99th percentiles).")

    df = _apply_stages(train_features, state, upto="fill")
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            median = df[column].median()
            state.numeric_fill[column] = float(median) if pd.notna(median) else 0.0

    df = _apply_stages(train_features, state, upto="encode")
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            continue
        values = _categorical_strings(df[column])
        if values.nunique(dropna=True) <= LOW_CARDINALITY_THRESHOLD:
            mode = values.dropna().mode()
            fill = str(mode.iloc[0]) if not mode.empty else "Unknown"
            categories = sorted(values.fillna(fill).unique().tolist())
            state.one_hot[column] = {"fill": fill, "categories": categories}
            steps.append(f"One-hot encoded column '{column}' into {len(categories)} features (unseen categories -> all zeros).")
        else:
            frequencies = values.fillna("Unknown").value_counts(normalize=True)
            state.frequency[column] = {"fill": "Unknown", "map": {str(k): float(v) for k, v in frequencies.items()}}
            steps.append(f"Frequency encoded high-cardinality column '{column}' (unseen categories -> 0).")

    state.feature_columns = list(_apply_stages(train_features, state).columns)
    return state


def _fit_target_mapping(target: pd.Series, task_type: str, mode: str) -> dict[str, int] | None:
    if task_type != "classification" or mode == "credit_scoring" or is_numeric_dtype(target):
        return None
    return {value: index for index, value in enumerate(sorted(target.astype(str).unique()))}


def _encode_target(target: pd.Series, task_type: str, mode: str, mapping: dict[str, int] | None) -> pd.Series:
    if mode == "credit_scoring":
        return pd.to_numeric(target).astype(int)
    if mapping is None:
        return target
    encoded = target.astype(str).map(mapping)
    if encoded.isna().any():
        unknown = sorted(target[encoded.isna()].astype(str).unique().tolist())
        raise ValueError(f"Validation labels not present in training rows: {', '.join(unknown)}.")
    return encoded.astype(int)


def _split_columns(df: pd.DataFrame, config: PrepConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    passthrough = config.passthrough_columns
    dropped = set(passthrough) | set(config.exclude_columns) | ({config.target_column} if config.target_column else set())
    features = df[[column for column in df.columns if column not in dropped]]
    target = df[config.target_column] if config.target_column else None
    return df[passthrough], features, target


def _run_eda(train_features: pd.DataFrame, config: PrepConfig) -> EDAReport:
    protected = [*config.passthrough_columns, *config.exclude_columns]
    if config.target_column:
        protected.append(config.target_column)
    # The agent only sees training features: no label, no IDs, no validation rows.
    return EDAAgent(
        train_features,
        target_column=None,
        use_llm=config.eda_use_llm,
        llm_strategy=config.eda_llm_strategy,
        max_llm_rounds=config.eda_llm_rounds,
        protected_columns=protected,
    ).run()


def prepare_split(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame | None,
    config: PrepConfig,
    task_type: str | None,
    eda_steps: list[FeatureEngineeringStep] | None = None,
) -> SplitResult:
    """Fit preprocessing on train_raw only, then transform both frames.

    This is the single code path used for the holdout split, every
    cross-validation fold, and the final full-data refit.
    """
    steps: list[str] = []
    content_columns = [column for column in train_raw.columns if column != SOURCE_ROW_COLUMN]
    duplicate_mask = train_raw.duplicated(subset=content_columns)
    duplicates = int(duplicate_mask.sum())
    if duplicates:
        train_raw = train_raw.loc[~duplicate_mask]
        steps.append(f"Removed {duplicates} duplicate rows from the training rows only.")

    train_pass, train_features, train_target = _split_columns(train_raw, config)

    eda_report = None
    if eda_steps is None:
        eda_steps = []
        if config.use_eda_agent:
            eda_report = _run_eda(train_features, config)
            eda_steps = list(eda_report.feature_engineering_steps)
            steps.append(
                "Applied EDA agent feature engineering with "
                f"{len(eda_steps)} generated steps chosen from training features only "
                f"(strategy={eda_report.llm_strategy}, validated_llm_steps={eda_report.validated_llm_steps})."
            )

    state = fit_preprocessor(train_features, config.ratio_pairs, eda_steps, steps)
    state.passthrough_columns = [column for column in config.passthrough_columns if column != SOURCE_ROW_COLUMN]
    state.target_column = config.target_column
    state.task_type = task_type

    if train_target is not None and task_type is not None:
        state.target_mapping = _fit_target_mapping(train_target, task_type, config.mode)
        if state.target_mapping is not None:
            steps.append(f"Encoded target column '{config.target_column}' into numeric class IDs (fitted on training labels).")
        train_target = _encode_target(train_target, task_type, config.mode, state.target_mapping)

    train_x, _ = transform_features(train_features, state)
    train_frame = SplitFrame(train_pass, train_x, train_target)

    valid_frame = None
    unseen: dict[str, int] = {}
    if valid_raw is not None:
        valid_pass, valid_features, valid_target = _split_columns(valid_raw, config)
        valid_x, unseen = transform_features(valid_features, state)
        if valid_target is not None and task_type is not None:
            valid_target = _encode_target(valid_target, task_type, config.mode, state.target_mapping)
        valid_frame = SplitFrame(valid_pass, valid_x, valid_target)

    return SplitResult(
        state=state,
        steps=steps,
        eda_report=eda_report,
        eda_steps=eda_steps,
        train=train_frame,
        valid=valid_frame,
        duplicate_rows_removed=duplicates,
        unseen_categories_in_valid=unseen,
    )


def _eda_applied_frame(frames: list[pd.DataFrame], config: PrepConfig, eda_steps: list[FeatureEngineeringStep]) -> pd.DataFrame:
    outputs = []
    for frame in frames:
        passthrough, features, target = _split_columns(frame, config)
        engineered = apply_feature_steps(_apply_ratios(features.copy(), config.ratio_pairs), eda_steps, strict=True)
        parts = [passthrough, engineered] + ([target] if target is not None else [])
        outputs.append(pd.concat(parts, axis=1))
    return pd.concat(outputs).sort_values(SOURCE_ROW_COLUMN).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_config(
    target_column: str | None = None,
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
) -> PrepConfig:
    return PrepConfig(
        target_column=target_column,
        task_type=task_type,
        mode=mode,
        split_strategy=split_strategy,
        group_column=group_column or None,
        time_column=time_column or None,
        id_columns=list(id_columns or []),
        exclude_columns=list(exclude_columns or []),
        ratio_features=list(ratio_features or []),
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        use_eda_agent=use_eda_agent,
        eda_use_llm=eda_use_llm,
        eda_llm_strategy=eda_llm_strategy,
        eda_llm_rounds=eda_llm_rounds,
    )


def prepare_dataset(
    csv_path: str,
    target_column: str | None = None,
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
) -> DatasetPreparationArtifacts:
    config = build_config(
        target_column=target_column,
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
    source_path, raw_df = load_raw_frame(csv_path)
    validate_config(config, raw_df)

    dataset_name = _slugify(source_path.stem)
    destination = Path(output_dir).expanduser().resolve() if output_dir else (source_path.parent / f"{dataset_name}_prepared")
    destination.mkdir(parents=True, exist_ok=True)

    source_df = raw_df.drop(columns=[SOURCE_ROW_COLUMN])
    # Profiles and column batches only organise the human EDA report. They are
    # never used to fit preprocessing and are not train/validation splits.
    source_profile = build_dataset_profile(source_df, config.target_column)
    work_queue = build_work_queue(source_profile)
    original_shape = {"rows": int(len(source_df)), "columns": int(len(source_df.columns))}

    task_type_resolved: str | None = None
    split_strategy_resolved: str | None = None
    steps: list[str] = []
    train_path: Path | None = None
    valid_path: Path | None = None
    split_sizes: dict[str, int] = {}

    if config.target_column:
        task_type_resolved = resolve_task_type(raw_df[config.target_column], config)
        train_pos, valid_pos, split_strategy_resolved = split_rows(raw_df, config, task_type_resolved)
        train_raw, valid_raw = raw_df.iloc[train_pos], raw_df.iloc[valid_pos]
        steps.append(
            f"Split raw rows before any fitting ({split_strategy_resolved}): "
            f"{len(train_raw)} train / {len(valid_raw)} validation."
        )
        split = prepare_split(train_raw, valid_raw, config, task_type_resolved)
        steps.extend(split.steps)

        train_df = split.train.combined()
        valid_df = split.valid.combined()  # type: ignore[union-attr]
        train_path = destination / f"{dataset_name}_prepared_train.csv"
        valid_path = destination / f"{dataset_name}_prepared_valid.csv"
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        split_sizes = {"train": int(len(train_df)), "valid": int(len(valid_df))}

        # Final full-data refit: a separate fit on every labelled row, for the
        # final model only. It is never used for validation scoring.
        full = prepare_split(raw_df, None, config, task_type_resolved, eda_steps=split.eda_steps)
        steps.append("Refitted preprocessing on all labelled rows for final training (not used for validation scores).")
        eda_frames = [train_raw, valid_raw]
    else:
        split = prepare_split(raw_df, None, config, None)
        steps.extend(split.steps)
        full = split
        eda_frames = [raw_df]

    full_df = full.train.combined()
    full_path = destination / f"{dataset_name}_prepared_full.csv"
    full_df.to_csv(full_path, index=False)
    preprocessor_path = split.state.save(destination / f"{dataset_name}_preprocessor.json")
    preprocessor_full_path = full.state.save(destination / f"{dataset_name}_preprocessor_full.json")

    eda_report = split.eda_report
    eda_artifacts: dict[str, str] = {}
    eda_applied_path: Path | None = None
    eda_applied_shape: dict[str, int] | None = None
    eda_added_columns: list[str] = []
    if eda_report is not None:
        eda_artifacts = write_eda_artifacts(eda_report, destination, dataset_name)
        eda_applied = _eda_applied_frame(eda_frames, config, split.eda_steps)
        eda_applied_path = destination / f"{dataset_name}_eda_applied.csv"
        eda_applied.to_csv(eda_applied_path, index=False)
        eda_applied_shape = {"rows": int(len(eda_applied)), "columns": int(len(eda_applied.columns))}
        eda_added_columns = [column for column in eda_applied.columns if column not in raw_df.columns]

    prepared_reference = split.train.combined() if config.target_column else full_df
    prepared_profile = build_dataset_profile(
        prepared_reference.drop(columns=config.passthrough_columns), config.target_column
    )
    feature_columns = split.state.feature_columns

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
        preparation_summary={"steps": steps, "feature_count": len(feature_columns)},
        evaluation_summary=None,
    )

    post_outcome_columns = find_post_outcome_columns(
        [column for column in source_df.columns if column not in config.exclude_columns and column != config.target_column]
    )
    if config.mode == "credit_scoring" and post_outcome_columns:
        steps.append(
            "Review possible post-loan outcome columns and exclude them with exclude_columns unless they are "
            f"known at prediction time: {', '.join(post_outcome_columns)}."
        )

    encoding_map = {column: "one_hot" for column in split.state.one_hot}
    encoding_map.update({column: "frequency" for column in split.state.frequency})

    summary = {
        "dataset_name": dataset_name,
        "source_path": str(source_path),
        "output_dir": str(destination),
        "target_column": config.target_column,
        "task_type": task_type_resolved,
        "mode": config.mode,
        "target_definition": CREDIT_TARGET_DEFINITION if config.mode == "credit_scoring" else None,
        "split_strategy": split_strategy_resolved,
        "validation_fraction": config.validation_fraction,
        "split_sizes": split_sizes,
        "group_column": config.group_column,
        "time_column": config.time_column,
        "id_columns": config.id_columns,
        "passthrough_columns": config.passthrough_columns,
        "excluded_columns": config.exclude_columns,
        "ratio_features": config.ratio_features,
        "possible_post_outcome_columns": post_outcome_columns,
        "duplicate_rows_removed_from_train": split.duplicate_rows_removed,
        "unseen_categories_in_valid": split.unseen_categories_in_valid,
        "target_mapping": split.state.target_mapping,
        "original_shape": original_shape,
        "prepared_shape": {"rows": int(len(full_df)), "columns": int(len(full_df.columns))},
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "full_feature_columns": full.state.feature_columns,
        "dropped_identifier_columns": split.state.identifier_columns,
        "encoded_columns": list(encoding_map),
        "encoding_map": encoding_map,
        "steps": steps,
        "preprocessor_path": str(preprocessor_path),
        "preprocessor_full_path": str(preprocessor_full_path),
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
        "eda_llm_strategy": eda_report.llm_strategy if eda_report else "none",
        "eda_llm_rounds_run": eda_report.llm_rounds_run if eda_report else 0,
        "eda_llm_rounds_requested": int(eda_llm_rounds),
        "eda_llm_candidate_steps": eda_report.llm_candidate_steps if eda_report else 0,
        "eda_llm_provider": eda_report.llm_provider if eda_report else "none",
        "eda_summary": eda_report.agent_summary if eda_report else None,
        "eda_recommendations": eda_report.agent_recommendations if eda_report else [],
        "eda_feature_engineering_steps": len(split.eda_steps),
        "eda_validated_llm_steps": eda_report.validated_llm_steps if eda_report else 0,
        "eda_rejected_llm_steps": eda_report.rejected_llm_steps if eda_report else 0,
        "eda_llm_rejection_reasons": eda_report.llm_rejection_reasons if eda_report else [],
        "eda_llm_round_records": eda_report.to_dict()["llm_round_records"] if eda_report else [],
        "eda_applied_path": str(eda_applied_path) if eda_applied_path else None,
        "eda_applied_shape": eda_applied_shape,
        "eda_added_columns": eda_added_columns,
        "eda_removed_columns": [],
        **eda_artifacts,
        "ready_for_training": True,
        "recommended_model_family": "tree_based_boosting_or_linear_baseline",
    }

    manifest_path = destination / f"{dataset_name}_feature_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, default=str))

    return DatasetPreparationArtifacts(
        summary=summary,
        prepared_full_path=full_path,
        prepared_train_path=train_path,
        prepared_valid_path=valid_path,
        manifest_path=manifest_path,
        config=config,
    )


def apply_preprocessor(csv_path: str, state_path: str, output_path: str) -> dict[str, Any]:
    """Transform new rows (e.g. a competition test file) with a saved state.

    Every input row is kept, in input order, with its ID columns alongside.
    """
    state = PreprocessorState.load(state_path)
    _, raw_df = load_raw_frame(csv_path)
    missing_ids = [column for column in state.passthrough_columns if column not in raw_df.columns]
    passthrough_columns = [SOURCE_ROW_COLUMN] + [c for c in state.passthrough_columns if c in raw_df.columns]
    features = raw_df.drop(columns=passthrough_columns)
    transformed, unseen = transform_features(features, state)
    output = pd.concat([raw_df[passthrough_columns].reset_index(drop=True), transformed], axis=1)
    if len(output) != len(raw_df):
        raise ValueError("Row count changed during transform.")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return {
        "output_path": str(destination),
        "rows": int(len(output)),
        "feature_count": len(state.feature_columns),
        "passthrough_columns": passthrough_columns,
        "passthrough_columns_not_found": missing_ids,
        "unseen_categories": unseen,
    }
