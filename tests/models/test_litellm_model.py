import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swe_core.agents.default import DefaultAgent
from swe_core.environments.local import LocalEnvironment
from swe_core.exceptions import FormatError
from swe_core.models.litellm_model import LitellmModel
from swe_core.models.utils.actions_toolcall import BASH_TOOL


def make_response(
    tool_calls: list[MagicMock] | None,
) -> MagicMock:
    response = MagicMock()

    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message.tool_calls = tool_calls
    choice.message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
    }

    response.choices = [choice]
    response.model_dump.return_value = {
        "id": "response-1",
    }

    return response


def make_tool_call(
    *,
    name: str = "bash",
    arguments: str = '{"command": "echo hello"}',
) -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = "call-1"
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


@patch(
    "swe_core.models.litellm_model."
    "litellm.cost_calculator.completion_cost",
    return_value=0.01,
)
@patch(
    "swe_core.models.litellm_model."
    "litellm.completion",
)
def test_query_returns_action_and_metadata(
    mock_completion: MagicMock,
    mock_cost: MagicMock,
) -> None:
    response = make_response([
        make_tool_call(),
    ])
    mock_completion.return_value = response

    model = LitellmModel(
        model_name="openai/test-model",
    )

    message = model.query([
        {
            "role": "user",
            "content": "say hello",
            "extra": {
                "internal": True,
            },
        },
    ])

    call_kwargs = mock_completion.call_args.kwargs

    assert call_kwargs["model"] == "openai/test-model"
    assert call_kwargs["tools"] == [BASH_TOOL]
    assert call_kwargs["messages"] == [
        {
            "role": "user",
            "content": "say hello",
        },
    ]

    assert message["extra"]["actions"] == [
        {
            "command": "echo hello",
            "tool_call_id": "call-1",
        },
    ]
    assert message["extra"]["cost"] == pytest.approx(
        0.01
    )
    assert message["extra"]["response"] == {
        "id": "response-1",
    }
    assert "timestamp" in message["extra"]


@patch(
    "swe_core.models.litellm_model."
    "litellm.cost_calculator.completion_cost",
    return_value=0.02,
)
@patch(
    "swe_core.models.litellm_model."
    "litellm.completion",
)
def test_format_error_preserves_cost_and_response(
    mock_completion: MagicMock,
    mock_cost: MagicMock,
) -> None:
    response = make_response(None)
    mock_completion.return_value = response

    model = LitellmModel(
        model_name="openai/test-model",
    )

    with pytest.raises(FormatError) as exc_info:
        model.query([
            {
                "role": "user",
                "content": "test",
            },
        ])

    extra = exc_info.value.messages[0]["extra"]

    assert extra["cost"] == pytest.approx(0.02)
    assert extra["response"] == {
        "id": "response-1",
    }


def test_format_observation_message() -> None:
    model = LitellmModel(
        model_name="openai/test-model",
        observation_template="{{ output.output }}",
    )

    messages = model.format_observation_messages(
        {
            "extra": {
                "actions": [
                    {
                        "command": "echo hello",
                        "tool_call_id": "call-1",
                    },
                ],
            },
        },
        [
            {
                "output": "hello",
                "returncode": 0,
            },
        ],
    )

    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call-1"
    assert messages[0]["content"] == "hello"


@patch(
    "swe_core.models.litellm_model."
    "litellm.cost_calculator.completion_cost",
    return_value=0.01,
)
@patch(
    "swe_core.models.litellm_model."
    "litellm.completion",
)
def test_default_agent_litellm_local_submission(
    mock_completion: MagicMock,
    mock_cost: MagicMock,
    tmp_path: Path,
) -> None:
    command = (
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        "\\nstage 7 complete\\n'"
    )
    response = make_response([
        make_tool_call(
            arguments=json.dumps({
                "command": command,
            }),
        ),
    ])
    mock_completion.return_value = response

    agent = DefaultAgent(
        model=LitellmModel(
            model_name="openai/test-model",
        ),
        env=LocalEnvironment(
            cwd=str(tmp_path),
        ),
        system_template="You are a coding agent.",
        instance_template="Task: {{ task }}",
        step_limit=2,
        cost_limit=0.0,
    )

    result = agent.run("submit the task")

    assert result == {
        "exit_status": "Submitted",
        "submission": "stage 7 complete\n",
    }
    assert agent.n_calls == 1
    assert agent.cost == pytest.approx(0.01)
    assert agent.messages[-1]["role"] == "exit"
    mock_completion.assert_called_once()
