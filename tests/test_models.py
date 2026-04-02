"""Unit tests for environment models."""

from environment.models import Action, Observation, Reward


def test_observation_serialization_round_trip() -> None:
    observation = Observation(
        task_id="null_filling",
        task_name="Missing Value Repair",
        task_description="Repair missing values in the contacts table.",
        step=0,
        max_steps=5,
        available_actions=["fill_missing"],
        context="Inspect the dataset preview.",
        dataset_preview="row_id | name | age\n0 | Alice |",
        issues_remaining=3,
        attempts_remaining=5,
        feedback="",
        feedback_history=[],
    )

    dumped = observation.model_dump(mode="json")
    restored = Observation.model_validate(dumped)

    assert restored.task_id == "null_filling"
    assert restored.attempts_remaining == 5
    assert restored.dataset_preview.startswith("row_id")


def test_action_validation_normalizes_action_type() -> None:
    action = Action(
        action_type="  FIX_VALUE  ",
        row_index=2,
        column="email",
        new_value="user@example.com",
        reason="Replace the missing email.",
    )
    assert action.action_type == "fix_value"


def test_action_from_llm_output_parses_json_block() -> None:
    action = Action.from_llm_output(
        '```json\n{"action_type":"fill_missing","row_index":2,"column":"email","new_value":"user@example.com","reason":"row 2 email is missing"}\n```'
    )
    assert action.action_type == "fill_missing"
    assert action.row_index == 2


def test_action_schema_enum_matches_runtime_validation() -> None:
    action_names = Action.model_json_schema()["properties"]["action_type"]["enum"]

    for action_name in action_names:
        action = Action(
            action_type=action_name,
            row_index=0,
            column="example_column",
            new_value="value",
            reason="Schema/runtime consistency check.",
        )
        assert action.action_type == action_name


def test_reward_clamps_value_and_fields() -> None:
    reward = Reward(value=1.4, issues_fixed_this_step=2, issues_remaining=0, solved=True, attempts_used=1)

    assert reward.value == 1.0
    assert reward.issues_fixed_this_step == 2
    assert reward.solved is True
