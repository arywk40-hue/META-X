"""Shared Pydantic models and task definitions for the code review environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "bug_line": {"type": "integer", "description": "Line number of the bug"},
        "bug_type": {
            "type": "string",
            "enum": ["syntax", "runtime", "logic", "security"],
        },
        "explanation": {"type": "string", "description": "Why this is a bug"},
    },
    "required": ["bug_line", "bug_type", "explanation"],
}


class Difficulty(str, Enum):
    """Difficulty bucket used by tasks and API payloads."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @property
    def level(self) -> int:
        return {
            Difficulty.EASY: 1,
            Difficulty.MEDIUM: 2,
            Difficulty.HARD: 3,
        }[self]


@dataclass(slots=True)
class Task:
    """Immutable task definition loaded at startup."""

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
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.target_score_range == (0.0, 1.0):
            self.target_score_range = {
                Difficulty.EASY: (0.7, 1.0),
                Difficulty.MEDIUM: (0.4, 0.7),
                Difficulty.HARD: (0.1, 0.4),
            }[self.difficulty]

    def summary(self) -> dict[str, Any]:
        """Return the public task descriptor without leaking the answer key."""
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
            "code_snippet": self.config.get("code_snippet", ""),
        }


class CodeReviewAction(BaseModel):
    """Structured bug report submitted by the agent."""

    bug_line: int = Field(..., ge=1, description="Line number where the bug appears")
    bug_type: str = Field(..., description="One of syntax, runtime, logic, security")
    explanation: str = Field(..., min_length=1, description="Concise explanation of the bug")

    @field_validator("bug_type")
    @classmethod
    def validate_bug_type(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"syntax", "runtime", "logic", "security"}:
            raise ValueError("bug_type must be one of syntax, runtime, logic, security")
        return cleaned

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("explanation cannot be empty")
        return cleaned

    @classmethod
    def from_llm_output(
        cls,
        text: str,
        available_actions: list[str] | None = None,
    ) -> "CodeReviewAction":
        """Parse flexible model output into a structured bug report."""
        candidate = text.strip()
        if not candidate:
            return cls(
                bug_line=1,
                bug_type="logic",
                explanation="No valid answer was produced.",
            )

        candidate = candidate.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        json_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            line_match = re.search(r"line\s*(\d+)", candidate, re.IGNORECASE)
            type_match = re.search(
                r"\b(syntax|runtime|logic|security)\b",
                candidate,
                re.IGNORECASE,
            )
            return cls(
                bug_line=int(line_match.group(1)) if line_match else 1,
                bug_type=type_match.group(1).lower() if type_match else "logic",
                explanation=candidate,
            )

        return cls.model_validate(data)

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: Any = None,
        mode: str = "validation",
    ) -> dict[str, Any]:
        """Return the exact public schema exposed at /tasks."""
        return dict(ACTION_SCHEMA)


class CodeReviewObservation(BaseModel):
    """Observation returned by reset() and step()."""

    done: bool = Field(default=False, description="Episode completion flag")
    code_snippet: str = Field(..., description="Python code snippet with line numbers")
    task_id: str = Field(..., description="Unique identifier for the current task")
    attempts_remaining: int = Field(..., ge=0, description="Remaining attempts before episode termination")
    feedback: str = Field(default="", description="Human-readable feedback for the last attempt")
    feedback_history: list[str] = Field(default_factory=list, description="All previous feedback messages")
    task_name: str = Field(..., description="Human-readable task name")
    task_description: str = Field(..., description="Full description of the task objective")
    step: int = Field(..., ge=0, description="Current step number in episode")
    max_steps: int = Field(..., gt=0, description="Maximum attempts allowed for this task")
    available_actions: list[str] = Field(default_factory=list, description="Valid action types for the current state")
    context: str = Field(default="", description="Human-readable context for the reviewer")
    timestamp: datetime = Field(default_factory=utc_now, description="Observation timestamp")

    def to_prompt(self) -> str:
        """Convert the observation into an LLM-friendly prompt."""
        history = "\n".join(self.feedback_history) if self.feedback_history else "None"
        return (
            f"Task: {self.task_name}\n"
            f"Description: {self.task_description}\n"
            f"Attempt: {self.step + 1} of {self.max_steps}\n"
            f"Attempts Remaining: {self.attempts_remaining}\n"
            f"Feedback History:\n{history}\n\n"
            f"Code Snippet:\n{self.code_snippet}\n\n"
            "Respond with JSON only: "
            '{"bug_line": <int>, "bug_type": "syntax|runtime|logic|security", "explanation": "<concise explanation>"}'
        )


class CodeReviewReward(BaseModel):
    """Dense reward returned after each environment step."""

    value: float = Field(..., ge=0.0, le=1.0, description="Reward normalized to [0, 1]")
    partial_credit: float = Field(..., ge=0.0, le=1.0, description="Score before attempt bonus")
    solved: bool = Field(..., description="Whether the current attempt solved the task")
    attempts_used: int = Field(..., ge=0, description="Number of attempts consumed in the episode")

    @field_validator("value", "partial_credit", mode="before")
    @classmethod
    def clamp_unit_interval(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class CodeReviewState(BaseModel):
    """Typed state snapshot returned by state()."""

    episode_id: str
    step_count: int = Field(..., ge=0)
    task_id: str
    solved: bool
    final_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("final_score", mode="before")
    @classmethod
    def clamp_final_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


Action = CodeReviewAction
Observation = CodeReviewObservation
Reward = CodeReviewReward
State = CodeReviewState


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action: Action


class GraderRequest(BaseModel):
    task_id: str
    episode: list[dict[str, Any]]


class BaselineRequest(BaseModel):
    task_ids: list[str] | None = None
    model: str = "gpt-4o-mini"
    max_episodes_per_task: int = Field(default=1, ge=1, le=10)
    verbose: bool = False
