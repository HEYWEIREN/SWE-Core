# tests/models/test_deterministic.py

from typing import Any

import pytest

from swe_core.environments.local import LocalEnvironment
from swe_core.models.test_models import (
    DeterministicModel,
    make_output,
)


def test_query_returns_outputs_in_order() -> None:
    first = make_output(
        "First command",
        [{"command": "echo first"}],
    )
    second = make_output(
        "Second command",
        [{"command": "echo second"}],
    )

    model = DeterministicModel(
        outputs=[first, second]
    )

    first_result = model.query([])
    second_result = model.query(
        [first_result]
    )

    assert first_result == first
    assert second_result == second


def test_query_raises_when_outputs_are_exhausted() -> None:
    model = DeterministicModel(
        outputs=[
            make_output(
                "Only command",
                [{"command": "echo only"}],
            )
        ]
    )

    model.query([])

    with pytest.raises(IndexError):
        model.query([])


def test_query_returns_configured_output() -> None:
    original = make_output(
        "Print hello",
        [{"command": "echo hello"}],
    )

    model = DeterministicModel(
        outputs=[original]
    )

    result = model.query([])

    assert result is model.config.outputs[0]


def test_format_message_preserves_fields() -> None:
    model = DeterministicModel(
        outputs=[]
    )

    message = model.format_message(
        role="exit",
        content="Submitted",
        extra={
            "exit_status": "Submitted",
            "submission": "finished",
        },
    )

    assert message == {
        "role": "exit",
        "content": "Submitted",
        "extra": {
            "exit_status": "Submitted",
            "submission": "finished",
        },
    }


def test_format_observation_messages() -> None:
    model = DeterministicModel(
        outputs=[]
    )

    assistant_message = make_output(
        "Print hello",
        [{"command": "echo hello"}],
    )

    outputs: list[dict[str, Any]] = [
        {
            "output": "hello\n",
            "returncode": 0,
            "exception_info": "",
        }
    ]

    observations = (
        model.format_observation_messages(
            assistant_message,
            outputs,
            {},
        )
    )

    assert len(observations) == 1

    observation = observations[0]

    assert observation["role"] == "user"
    assert (
        "<returncode>0</returncode>"
        in observation["content"]
    )
    assert "hello" in observation["content"]

    assert observation["extra"]["raw_output"] == "hello\n"
    assert observation["extra"]["returncode"] == 0
    assert observation["extra"]["exception_info"] == ""
    assert "timestamp" in observation["extra"]


def test_model_environment_round_trip() -> None:
    """
    手动完成一轮：

    initial messages
        -> model.query()
        -> Environment.execute()
        -> observation messages

    第 5 阶段会把这部分逻辑放进 DefaultAgent。
    """

    model = DeterministicModel(
        outputs=[
            make_output(
                "Print hello",
                [{"command": "printf hello"}],
            )
        ]
    )

    environment = LocalEnvironment()

    messages = [
        model.format_message(
            role="system",
            content="You are a coding agent.",
        ),
        model.format_message(
            role="user",
            content="Print hello.",
        ),
    ]

    assistant_message = model.query(
        messages
    )
    messages.append(assistant_message)

    actions = assistant_message.get(
        "extra",
        {},
    ).get(
        "actions",
        [],
    )

    outputs = [
        environment.execute(action)
        for action in actions
    ]

    observations = (
        model.format_observation_messages(
            assistant_message,
            outputs,
            {},
        )
    )

    messages.extend(observations)

    assert [
        message["role"]
        for message in messages
    ] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    assert (
        assistant_message["extra"]["actions"]
        == [{"command": "printf hello"}]
    )

    assert (
        messages[-1]["extra"]["returncode"]
        == 0
    )
    assert "hello" in messages[-1]["content"]
