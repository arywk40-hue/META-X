"""Unit tests for deterministic graders."""

from environment import Action, grade_episode


def _play_perfect_runtime_episode(env) -> list[dict]:
    env.reset("null_filling")
    env.step(
        Action(
            action_type="fill_missing",
            row_index=1,
            column="age",
            new_value="unknown",
            reason="The age value is missing.",
        )
    )
    env.step(
        Action(
            action_type="fix_value",
            row_index=2,
            column="age",
            new_value="unknown",
            reason="The literal NULL should be normalized.",
        )
    )
    env.step(
        Action(
            action_type="fill_missing",
            row_index=4,
            column="email",
            new_value="missing@example.com",
            reason="The email value is missing.",
        )
    )
    return env.state()["episode_history"]


def test_all_graders_return_values_in_range(mock_tasks) -> None:
    for task_id in mock_tasks:
        score = grade_episode(task_id, [])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_perfect_episode_scores_one(sample_env) -> None:
    history = _play_perfect_runtime_episode(sample_env)
    assert grade_episode("null_filling", history) == 1.0


def test_grader_is_deterministic(sample_env) -> None:
    history = _play_perfect_runtime_episode(sample_env)
    first = grade_episode("null_filling", history)
    second = grade_episode("null_filling", history)
    assert first == second


def test_grader_has_multiple_distinct_score_levels() -> None:
    wrong = [{"action": {"action_type": "flag_anomaly", "row_index": 0, "column": "status", "new_value": None, "reason": "Totally wrong"}}]
    partial = [{"action": {"action_type": "drop_row", "row_index": 1, "column": "row_id", "new_value": None, "reason": "Duplicate row"}}]
    stronger_partial = [
        {
            "action": {
                "action_type": "drop_row",
                "row_index": 1,
                "column": "row_id",
                "new_value": None,
                "reason": "Duplicate row",
            }
        },
        {
            "action": {
                "action_type": "flag_anomaly",
                "row_index": 2,
                "column": "amount",
                "new_value": None,
                "reason": "Extreme outlier",
            }
        },
    ]
    perfect = [
        {
            "action": {
                "action_type": "drop_row",
                "row_index": 1,
                "column": "row_id",
                "new_value": None,
                "reason": "Duplicate row",
            }
        },
        {
            "action": {
                "action_type": "flag_anomaly",
                "row_index": 2,
                "column": "amount",
                "new_value": None,
                "reason": "Extreme outlier",
            }
        },
        {
            "action": {
                "action_type": "fix_value",
                "row_index": 3,
                "column": "amount",
                "new_value": 50.0,
                "reason": "Negative amount must be non-negative",
            }
        },
        {
            "action": {
                "action_type": "standardize_format",
                "row_index": 5,
                "column": "status",
                "new_value": "completed",
                "reason": "Normalize status casing",
            }
        },
    ]

    scores = {
        grade_episode("duplicate_outlier", wrong),
        grade_episode("duplicate_outlier", partial),
        grade_episode("duplicate_outlier", stronger_partial),
        grade_episode("duplicate_outlier", perfect),
    }

    assert len(scores) >= 4
