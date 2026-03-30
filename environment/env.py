"""Core OpenEnv implementation for the code review domain."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from datetime import datetime, timezone
import random
import threading
from typing import Any
from uuid import uuid4

from .graders import grade_episode, score_bug_report
from .models import Action, Observation, Reward, State, Task
from .reward import compute_reward
from .tasks import TASKS, get_task


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OpenEnvBase(ABC):
    """Abstract base class for OpenEnv-compatible environments."""

    @abstractmethod
    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> dict[str, Any]:
        raise NotImplementedError


class OpenEnv(OpenEnvBase):
    """Stateful environment for iterative Python code review."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.current_task: Task | None = None
        self.episode_id: str | None = None
        self.episode_history: list[dict[str, Any]] = []
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.current_observation: Observation | None = None
        self.started_at: datetime | None = None
        self.last_action_at: datetime | None = None
        self.seed: int | None = None
        self.task_state: dict[str, Any] = {}

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        with self._lock:
            task = get_task(task_id) if task_id else self._choose_task(seed)
            self.current_task = task
            self.episode_id = str(uuid4())
            self.seed = seed
            self.started_at = utc_now()
            self.last_action_at = None
            self.step_count = 0
            self.done = False
            self.cumulative_reward = 0.0
            self.episode_history = []
            self.task_state = {
                "attempts_remaining": task.max_steps,
                "feedback_history": [],
                "feedback": "",
                "solved": False,
                "final_score": 0.0,
            }
            self.current_observation = self._build_observation()
            return self.current_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        with self._lock:
            if self.current_task is None:
                raise ValueError("No active episode. Call reset() first.")
            if self.done:
                raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

            state_before = self._reward_state()
            observation_before = self.current_observation or self._build_observation()
            transition = self._evaluate_action(action)

            self.step_count += 1
            self.last_action_at = utc_now()
            self.task_state["attempts_remaining"] = max(0, self.current_task.max_steps - self.step_count)
            self.task_state["feedback"] = transition["feedback"]
            self.task_state["feedback_history"].append(transition["feedback"])
            self.task_state["solved"] = transition["solved"]
            self.done = transition["solved"] or self.task_state["attempts_remaining"] == 0

            state_after = self._reward_state(transition=transition)
            reward = compute_reward(state_before, action, state_after, task_config=self.current_task.config)
            self.cumulative_reward += reward.value

            info = self._build_info(transition)
            history_entry = {
                "step": self.step_count,
                "observation": observation_before.model_dump(mode="json"),
                "action": action.model_dump(mode="json"),
                "reward": reward.value,
                "reward_detail": reward.model_dump(mode="json"),
                "done": self.done,
                "info": info,
            }
            self.episode_history.append(history_entry)

            if self.done:
                final_score = grade_episode(self.current_task.id, self.episode_history)
                self.task_state["final_score"] = final_score
                info["final_score"] = final_score

            self.current_observation = self._build_observation()
            return self.current_observation, reward, self.done, info

    def state(self) -> dict[str, Any]:
        with self._lock:
            if self.current_task is None or self.current_observation is None or self.episode_id is None:
                raise ValueError("No active episode. Call reset() to start.")

            state = State(
                episode_id=self.episode_id,
                step_count=self.step_count,
                task_id=self.current_task.id,
                solved=bool(self.task_state.get("solved", False)),
                final_score=float(self.task_state.get("final_score", 0.0)),
            )

            payload = state.model_dump(mode="json")
            payload.update(
                {
                    "task_name": self.current_task.name,
                    "difficulty": self.current_task.difficulty.value,
                    "done": self.done,
                    "attempts_remaining": self.task_state.get("attempts_remaining", 0),
                    "cumulative_reward": round(self.cumulative_reward, 4),
                    "current_observation": self.current_observation.model_dump(mode="json"),
                    "episode_history": copy.deepcopy(self.episode_history),
                    "metadata": {
                        "started_at": self.started_at.isoformat() if self.started_at else None,
                        "last_action_at": self.last_action_at.isoformat() if self.last_action_at else None,
                        "seed": self.seed,
                    },
                }
            )
            return payload

    def _choose_task(self, seed: int | None) -> Task:
        chooser = random.Random(seed)
        return chooser.choice(list(TASKS.values()))

    def _evaluate_action(self, action: Action) -> dict[str, Any]:
        answer = self.current_task.config["answer"]
        step_number = self.step_count + 1
        reward_value, partial_credit, breakdown = score_bug_report(
            action.model_dump(mode="json"),
            answer,
            step_number,
        )
        solved = reward_value >= 0.85

        if solved:
            feedback = "Correct! You identified the bug."
        elif action.bug_line == answer["bug_line"]:
            feedback = "Right line, wrong type. Think about what category of error this is."
        elif action.bug_type == answer["bug_type"]:
            feedback = "Right bug type, wrong line. Look more carefully at the control flow."
        else:
            feedback = "Not quite. Re-read the function logic carefully."

        return {
            "reward_value": reward_value,
            "partial_credit": partial_credit,
            "feedback": feedback,
            "solved": solved,
            "breakdown": breakdown,
        }

    def _build_observation(self) -> Observation:
        task = self.current_task
        if task is None:
            raise RuntimeError("Environment has not been reset.")

        attempts_remaining = int(self.task_state.get("attempts_remaining", task.max_steps))
        feedback = str(self.task_state.get("feedback", ""))
        feedback_history = list(self.task_state.get("feedback_history", []))

        if self.done:
            context = f"Episode complete. Final score: {self.task_state.get('final_score', 0.0):.2f}."
        else:
            context = (
                "Review the Python code, identify the single most important bug, "
                "and cite the exact line, bug type, and explanation."
            )

        return Observation(
            done=self.done,
            code_snippet=task.config["code_snippet"],
            task_id=task.id,
            attempts_remaining=attempts_remaining,
            feedback=feedback,
            feedback_history=feedback_history,
            task_name=task.name,
            task_description=task.description.strip(),
            step=self.step_count,
            max_steps=task.max_steps,
            available_actions=list(task.config.get("available_actions", [])) if not self.done else [],
            context=context,
        )

    def _reward_state(self, transition: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "task_id": self.current_task.id if self.current_task else None,
            "step_count": self.step_count,
            "max_steps": self.current_task.max_steps if self.current_task else 0,
            "done": self.done,
            "last_transition": transition or {},
        }

    def _build_info(self, transition: dict[str, Any]) -> dict[str, Any]:
        return {
            "attempts_remaining": self.task_state.get("attempts_remaining", 0),
            "episode_progress": round(self.step_count / max(1, self.current_task.max_steps), 3),
            "solved": transition["solved"],
            "feedback": transition["feedback"],
            "partial_credit": transition["partial_credit"],
            "score_breakdown": transition["breakdown"],
        }
