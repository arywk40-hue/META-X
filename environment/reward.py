"""Dense per-step reward for the data-cleaning environment."""

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
    transition = next_state.get("last_transition", {})
    issues_fixed = int(transition.get("issues_fixed_this_step", 0))
    issues_remaining = int(transition.get("issues_remaining", 0))
    was_trap = bool(transition.get("trap_penalty", False))
    was_redundant = bool(transition.get("redundant_action", False))
    solved = bool(transition.get("solved", False))
    total_issues = int((task_config or {}).get("total_issues", 1))
    steps_used = int(next_state.get("step_count", 1))
    max_steps = int(next_state.get("max_steps", 1))

    reward = 0.0
    reward += issues_fixed * (0.2 / max(1, total_issues / 3))
    if was_trap:
        reward -= 0.1
    if was_redundant:
        reward -= 0.05
    if solved and steps_used <= max_steps // 2:
        reward += 0.15

    value = max(0.0, min(1.0, reward))
    return Reward(
        value=round(value, 4),
        issues_fixed_this_step=issues_fixed,
        issues_remaining=issues_remaining,
        solved=solved,
        attempts_used=steps_used,
    )
