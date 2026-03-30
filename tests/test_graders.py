"""Unit tests for deterministic graders."""

from environment import Action, grade_episode


def _play_perfect_runtime_episode(env) -> list[dict]:
    env.reset("runtime_bug")
    env.step(
        Action(
            bug_line=6,
            bug_type="runtime",
            explanation="The empty list case causes division by zero when len(numbers) is zero.",
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
    assert grade_episode("runtime_bug", history) == 1.0


def test_grader_is_deterministic(sample_env) -> None:
    history = _play_perfect_runtime_episode(sample_env)
    first = grade_episode("runtime_bug", history)
    second = grade_episode("runtime_bug", history)
    assert first == second


def test_grader_has_multiple_distinct_score_levels() -> None:
    wrong = [{"action": {"bug_line": 1, "bug_type": "syntax", "explanation": "Totally wrong"}}]
    partial = [{"action": {"bug_line": 23, "bug_type": "logic", "explanation": "Wrong category"}}]
    stronger_partial = [
        {
            "action": {
                "bug_line": 23,
                "bug_type": "security",
                "explanation": "This endpoint is unsafe and should be reviewed carefully.",
            }
        }
    ]
    perfect = [
        {
            "action": {
                "bug_line": 23,
                "bug_type": "security",
                "explanation": "This is SQL injection because the query uses an f-string instead of a parameterized query.",
            }
        }
    ]

    scores = {
        grade_episode("security_vulnerability", wrong),
        grade_episode("security_vulnerability", partial),
        grade_episode("security_vulnerability", stronger_partial),
        grade_episode("security_vulnerability", perfect),
    }

    assert len(scores) >= 4
