"""Unit tests for the core environment engine."""

import pytest

from environment import Action, Observation
from environment.tasks import TASKS


def test_reset_returns_valid_observation(sample_env) -> None:
    observation = sample_env.reset("null_filling")
    assert isinstance(observation, Observation)
    assert observation.task_id == "null_filling"
    assert observation.step == 0
    assert observation.attempts_remaining == 5


def test_step_advances_state_and_returns_reward(sample_env) -> None:
    starting_observation = sample_env.reset("null_filling")
    action = Action(
        action_type="fill_missing",
        row_index=1,
        column="age",
        new_value="unknown",
        reason="The age value is missing.",
    )
    observation, reward, done, info = sample_env.step(action)
    state = sample_env.state()

    assert state["step_count"] == 1
    assert reward.value > 0.0
    assert done is False
    assert info["solved"] is False
    assert observation.step == 1
    assert observation.issues_remaining == 2
    assert observation.dataset_preview != starting_observation.dataset_preview
    assert "unknown" in observation.dataset_preview


def test_drop_row_updates_dataset_preview(sample_env) -> None:
    sample_env.reset("duplicate_outlier")
    observation, reward, done, info = sample_env.step(
        Action(
            action_type="drop_row",
            row_index=1,
            column="row_id",
            new_value=None,
            reason="This row is an exact duplicate.",
        )
    )

    assert reward.value > 0.0
    assert done is False
    assert "<DROPPED>" in observation.dataset_preview


def test_state_requires_active_episode(sample_env) -> None:
    with pytest.raises(ValueError):
        sample_env.state()


def test_done_episode_requires_reset(sample_env) -> None:
    sample_env.reset("adversarial_sensor")
    sample_env.step(
        Action(
            action_type="fill_missing",
            row_index=2,
            column="reading",
            new_value=None,
            reason="The reading is missing.",
        )
    )
    sample_env.step(
        Action(
            action_type="flag_anomaly",
            row_index=5,
            column="reading",
            new_value=None,
            reason="999C is an impossible outlier.",
        )
    )

    with pytest.raises(RuntimeError):
        sample_env.step(
            Action(
                action_type="flag_anomaly",
                row_index=0,
                column="reading",
                new_value=None,
                reason="Trying again after the episode ended.",
            )
        )


def test_reset_without_task_id_picks_a_registered_task(sample_env) -> None:
    observation = sample_env.reset(seed=7)
    assert observation.task_id in TASKS


def test_episode_terminates_after_three_attempts(sample_env) -> None:
    observation = sample_env.reset("null_filling")
    done = False
    while not done:
        observation, reward, done, info = sample_env.step(
            Action(
                action_type="flag_anomaly",
                row_index=0,
                column="name",
                new_value=None,
                reason="This is not a real issue.",
            )
        )

    state = sample_env.state()
    assert state["done"] is True
    assert state["step_count"] == 5
    assert state["attempts_remaining"] == 0
