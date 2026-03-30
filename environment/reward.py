"""Dense reward calculation for the code review domain."""

from __future__ import annotations

from typing import Any

from .models import Action, Reward


def compute_reward(
    state: dict[str, Any],
    action: Action,
    next_state: dict[str, Any],
    task_config: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> Reward:
    """Build a reward object from the transition computed in env.step()."""
    transition = next_state.get("last_transition", {})
    reward_value = float(transition.get("reward_value", 0.0))
    partial_credit = float(transition.get("partial_credit", 0.0))
    solved = bool(transition.get("solved", False))
    attempts_used = int(next_state.get("step_count", 0))

    return Reward(
        value=reward_value,
        partial_credit=partial_credit,
        solved=solved,
        attempts_used=attempts_used,
    )
