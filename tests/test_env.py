"""Unit tests for the core environment engine."""

import pytest

from environment import Action, Observation


def test_reset_returns_valid_observation(sample_env) -> None:
    observation = sample_env.reset("runtime_bug")
    assert isinstance(observation, Observation)
    assert observation.task_id == "runtime_bug"
    assert observation.step == 0
    assert observation.attempts_remaining == 3


def test_step_advances_state_and_returns_reward(sample_env) -> None:
    sample_env.reset("runtime_bug")
    action = Action(
        bug_line=6,
        bug_type="runtime",
        explanation="The empty list case causes division by zero.",
    )
    observation, reward, done, info = sample_env.step(action)
    state = sample_env.state()

    assert state["step_count"] == 1
    assert reward.value >= 0.85
    assert done is True
    assert info["solved"] is True
    assert observation.step == 1


def test_state_requires_active_episode(sample_env) -> None:
    with pytest.raises(ValueError):
        sample_env.state()


def test_done_episode_requires_reset(sample_env) -> None:
    sample_env.reset("runtime_bug")
    sample_env.step(
        Action(
            bug_line=6,
            bug_type="runtime",
            explanation="The code divides by len(numbers) on an empty list, causing division by zero.",
        )
    )

    with pytest.raises(RuntimeError):
        sample_env.step(
            Action(
                bug_line=6,
                bug_type="runtime",
                explanation="Trying again after the episode ended.",
            )
        )


def test_reset_without_task_id_picks_a_registered_task(sample_env) -> None:
    observation = sample_env.reset(seed=7)
    assert observation.task_id in {"runtime_bug", "binary_search_logic", "security_vulnerability"}


def test_episode_terminates_after_three_attempts(sample_env) -> None:
    observation = sample_env.reset("binary_search_logic")
    done = False
    while not done:
        observation, reward, done, info = sample_env.step(
            Action(
                bug_line=1,
                bug_type="syntax",
                explanation="This is not the real bug.",
            )
        )

    state = sample_env.state()
    assert state["done"] is True
    assert state["step_count"] == 3
    assert state["attempts_remaining"] == 0
