"""Pydantic models for the data-cleaning environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [
                "fix_value",
                "drop_row",
                "fill_missing",
                "cast_type",
                "rename_column",
                "flag_anomaly",
                "standardize_format",
            ],
        },
        "row_index": {
            "type": "integer",
            "description": "0-based row index (null for column-level actions)",
        },
        "column": {"type": "string", "description": "Column name to act on"},
        "new_value": {"description": "Replacement value (string, number, or null)"},
        "reason": {"type": "string", "description": "One-sentence justification"},
    },
    "required": ["action_type", "column", "reason"],
}


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @property
    def level(self) -> int:
        return {"easy": 1, "medium": 2, "hard": 3}[self.value]


@dataclass(slots=True)
class Task:
    id: str
    name: str
    description: str
    difficulty: Difficulty
    max_steps: int
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    target_score_range: tuple[float, float] = (0.0, 1.0)
    success_criteria: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task id cannot be empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.target_score_range == (0.0, 1.0):
            self.target_score_range = {
                Difficulty.EASY: (0.7, 1.0),
                Difficulty.MEDIUM: (0.4, 0.7),
                Difficulty.HARD: (0.1, 0.4),
            }[self.difficulty]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.id,
            "name": self.name,
            "description": self.description.strip(),
            "difficulty": self.difficulty.value,
            "difficulty_level": self.difficulty.level,
            "max_steps": self.max_steps,
            "target_score_range": list(self.target_score_range),
            "tags": self.tags,
            "success_criteria": self.success_criteria,
            "dataset_preview": self.config.get("dataset_preview", ""),
        }


class DataCleaningAction(BaseModel):
    action_type: str
    row_index: int | None = Field(default=None, ge=0)
    column: str
    new_value: Any = None
    reason: str = Field(..., min_length=1)

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        aliases = {
            "fill_value": "fix_value",
            "replace_value": "fix_value",
            "mark_anomaly": "flag_anomaly",
            "standardize": "standardize_format",
        }
        valid = {
            "fix_value",
            "drop_row",
            "fill_missing",
            "cast_type",
            "rename_column",
            "flag_anomaly",
            "standardize_format",
        }
        cleaned = aliases.get(v.strip().lower(), v.strip().lower())
        if cleaned not in valid:
            raise ValueError(f"action_type must be one of {sorted(valid)}")
        return cleaned

    @field_validator("column")
    @classmethod
    def validate_column(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("column cannot be empty")
        return cleaned

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason cannot be empty")
        return v.strip()

    @classmethod
    def from_llm_output(
        cls,
        text: str,
        available_actions: list[str] | None = None,
    ) -> "DataCleaningAction":
        candidate = text.strip()
        if not candidate:
            return cls(
                action_type="flag_anomaly",
                column="unknown",
                reason="No valid action produced.",
            )

        candidate = candidate.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)

        try:
            data = json.loads(candidate)
            return cls.model_validate(data)
        except Exception:
            return cls(
                action_type="flag_anomaly",
                column="unknown",
                reason=candidate[:200],
            )

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: Any = None,
        mode: str = "validation",
    ) -> dict[str, Any]:
        return dict(ACTION_SCHEMA)


class DataCleaningObservation(BaseModel):
    done: bool = False
    task_id: str
    task_name: str
    task_description: str
    dataset_preview: str
    issues_remaining: int = 0
    step: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    attempts_remaining: int = Field(..., ge=0)
    feedback: str = ""
    feedback_history: list[str] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    context: str = ""
    timestamp: datetime = Field(default_factory=utc_now)

    def to_prompt(self) -> str:
        history = "\n".join(self.feedback_history) if self.feedback_history else "None"
        return (
            f"Task: {self.task_name}\n"
            f"Description: {self.task_description}\n"
            f"Step: {self.step + 1} of {self.max_steps} | "
            f"Issues remaining: {self.issues_remaining}\n"
            f"Previous feedback:\n{history}\n\n"
            f"Current dataset:\n{self.dataset_preview}\n\n"
            f"Available action types: {', '.join(self.available_actions)}\n\n"
            "Respond with valid JSON only — no prose, no markdown:\n"
            '{"action_type": "fix_value", "row_index": 2, "column": "email", '
            '"new_value": "user@example.com", "reason": "row 2 email is missing"}'
        )


class DataCleaningReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    issues_fixed_this_step: int = Field(default=0, ge=0)
    issues_remaining: int = Field(default=0, ge=0)
    solved: bool = False
    attempts_used: int = Field(..., ge=0)

    @field_validator("value", mode="before")
    @classmethod
    def clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class DataCleaningState(BaseModel):
    episode_id: str
    step_count: int = Field(..., ge=0)
    task_id: str
    solved: bool
    final_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("final_score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


Action = DataCleaningAction
Observation = DataCleaningObservation
Reward = DataCleaningReward
State = DataCleaningState


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class GraderRequest(BaseModel):
    task_id: str
    episode: list[dict[str, Any]]


class BaselineRequest(BaseModel):
    task_ids: list[str] | None = None
    model: str = "gpt-4o-mini"
    max_episodes_per_task: int = Field(default=1, ge=1, le=10)
    verbose: bool = False


class DatasetPreparationRequest(BaseModel):
    csv_path: str
    target_column: str | None = None
    output_dir: str | None = None
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    random_seed: int = Field(default=42)
    use_eda_agent: bool = False
    eda_use_llm: bool = False


class DatasetEvaluationRequest(BaseModel):
    csv_path: str
    target_column: str
    output_dir: str | None = None
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    random_seed: int = Field(default=42)
    use_eda_agent: bool = False
    eda_use_llm: bool = False
