"""Unit tests for environment models."""

from environment.models import Action, Observation, Reward


def test_observation_serialization_round_trip() -> None:
    observation = Observation(
        task_id="runtime_bug",
        task_name="Runtime Bug Review",
        task_description="Review the helper and find the bug.",
        step=0,
        max_steps=3,
        available_actions=["report_bug"],
        context="Review the snippet.",
        code_snippet="1  def calculate_average(numbers):\n2      return 0",
        attempts_remaining=3,
        feedback="",
        feedback_history=[],
    )

    dumped = observation.model_dump(mode="json")
    restored = Observation.model_validate(dumped)

    assert restored.task_id == "runtime_bug"
    assert restored.attempts_remaining == 3
    assert restored.code_snippet.startswith("1  def")


def test_action_validation_normalizes_bug_type() -> None:
    action = Action(
        bug_line=23,
        bug_type="  SECURITY  ",
        explanation="The query is vulnerable to SQL injection.",
    )
    assert action.bug_type == "security"


def test_action_from_llm_output_parses_json_block() -> None:
    action = Action.from_llm_output(
        '```json\n{"bug_line": 9, "bug_type": "logic", "explanation": "The loop condition skips the last candidate."}\n```'
    )
    assert action.bug_line == 9
    assert action.bug_type == "logic"


def test_reward_clamps_value_and_partial_credit() -> None:
    reward = Reward(value=1.4, partial_credit=1.2, solved=True, attempts_used=1)

    assert reward.value == 1.0
    assert reward.partial_credit == 1.0
    assert reward.solved is True
