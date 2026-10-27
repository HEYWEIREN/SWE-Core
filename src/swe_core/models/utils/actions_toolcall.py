# src/swe_core/models/utils/actions_toolcall.py

import json
import time
from typing import Any

from jinja2 import StrictUndefined, Template

from swe_core.exceptions import FormatError


BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def parse_toolcall_actions(
    tool_calls: list[Any] | None,
    *,
    format_error_template: str,
    template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    把模型返回的 tool calls 转换为 Agent 可以执行的 actions。

    成功时返回：
    [
        {
            "command": "echo hello",
            "tool_call_id": "call_123",
        }
    ]

    格式错误时抛出 FormatError，并携带一条用于纠正模型的用户消息。
    """

    template_kwargs = template_kwargs or {}

    if not tool_calls:
        message = _make_format_error_message(
            format_error_template=format_error_template,
            error=(
                "No tool calls found in the response. "
                "Every response MUST include at least one tool call."
            ),
            has_tool_calls=False,
            template_kwargs=template_kwargs,
        )

        raise FormatError(message)

    actions: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        error_message = ""
        arguments: Any = {}

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )
        except Exception as error:
            error_message += (
                "Error parsing tool call arguments: "
                f"{error}."
            )

        if tool_call.function.name != "bash":
            error_message += (
                f" Unknown tool "
                f"'{tool_call.function.name}'."
            )

        if not isinstance(arguments, dict):
            error_message += (
                " Tool call arguments must be "
                "a JSON object."
            )
        elif "command" not in arguments:
            error_message += (
                " Missing 'command' argument "
                "in bash tool call."
            )
        elif not isinstance(
            arguments["command"],
            str,
        ):
            error_message += (
                " The 'command' argument "
                "must be a string."
            )
        elif not arguments["command"].strip():
            error_message += (
                " The 'command' argument "
                "must not be empty."
            )

        if error_message:
            message = _make_format_error_message(
                format_error_template=format_error_template,
                error=error_message.strip(),
                has_tool_calls=True,
                template_kwargs=template_kwargs,
            )

            raise FormatError(message)

        actions.append(
            {
                "command": arguments["command"],
                "tool_call_id": tool_call.id,
            }
        )

    return actions


def _make_format_error_message(
    *,
    format_error_template: str,
    error: str,
    has_tool_calls: bool,
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    构造一条反馈给模型的格式纠错消息。
    """

    content = Template(
        format_error_template,
        undefined=StrictUndefined,
    ).render(
        error=error,
        actions=[],
        has_tool_calls=has_tool_calls,
        **template_kwargs,
    )

    return {
        "role": "user",
        "content": content,
        "extra": {
            "interrupt_type": "FormatError",
        },
    }


def format_toolcall_observation_messages(
    *,
    actions: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    observation_template: str,
    template_vars: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    把环境执行结果转换为 tool observation 消息。

    每个 action 与一个 output 对应。
    """

    template_vars = template_vars or {}

    not_executed = {
        "output": "",
        "returncode": -1,
        "exception_info": "action was not executed",
    }

    missing_count = max(
        0,
        len(actions) - len(outputs),
    )

    padded_outputs = (
        outputs
        + [not_executed] * missing_count
    )

    messages: list[dict[str, Any]] = []

    for action, output in zip(
        actions,
        padded_outputs,
    ):
        content = Template(
            observation_template,
            undefined=StrictUndefined,
        ).render(
            output=output,
            **template_vars,
        )

        message: dict[str, Any] = {
            "content": content,
            "extra": {
                "raw_output": output.get(
                    "output",
                    "",
                ),
                "returncode": output.get(
                    "returncode"
                ),
                "exception_info": output.get(
                    "exception_info"
                ),
                "timestamp": time.time(),
                **output.get("extra", {}),
            },
        }

        if "tool_call_id" in action:
            message["role"] = "tool"
            message["tool_call_id"] = (
                action["tool_call_id"]
            )
        else:
            # 兼容人工直接下发 action 的情况
            message["role"] = "user"

        messages.append(message)

    return messages
