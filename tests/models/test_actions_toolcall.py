from types import SimpleNamespace

import pytest

from swe_core.exceptions import FormatError
from swe_core.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


def tool_call(
    arguments: str = '{"command": "echo hello"}',
    *,
    name: str = "bash",
    call_id: str = "call_1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_bash_tool_schema() -> None:
    function = BASH_TOOL["function"]
    assert function["name"] == "bash"
    assert function["parameters"]["required"] == ["command"]


def test_parse_toolcall_actions() -> None:
    assert parse_toolcall_actions(
        [tool_call()],
        format_error_template="{{ error }}",
    ) == [{"command": "echo hello", "tool_call_id": "call_1"}]


@pytest.mark.parametrize(
    ("calls", "error"),
    [
        ([], "No tool calls found"),
        ([tool_call("not json")], "Error parsing tool call arguments"),
        ([tool_call(name="python")], "Unknown tool 'python'"),
        ([tool_call("{}")], "Missing 'command' argument"),
    ],
)
def test_invalid_toolcalls_raise_format_error(
    calls: list[SimpleNamespace],
    error: str,
) -> None:
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions(calls, format_error_template="{{ error }}")

    message = exc_info.value.messages[0]
    assert error in message["content"]
    assert message["extra"]["interrupt_type"] == "FormatError"


def test_format_error_template_receives_context() -> None:
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions(
            [],
            format_error_template="{{ finish_reason }}",
            template_kwargs={"finish_reason": "length"},
        )

    assert exc_info.value.messages[0]["content"] == "length"


def test_format_toolcall_observations() -> None:
    messages = format_toolcall_observation_messages(
        actions=[
            {"command": "first", "tool_call_id": "call_1"},
            {"command": "second", "tool_call_id": "call_2"},
        ],
        outputs=[{"output": "done", "returncode": 0}],
        observation_template="{{ label }}:{{ output.output }}",
        template_vars={"label": "result"},
    )

    assert messages[0]["content"] == "result:done"
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_1"
    assert messages[1]["tool_call_id"] == "call_2"
    assert messages[1]["extra"]["returncode"] == -1
