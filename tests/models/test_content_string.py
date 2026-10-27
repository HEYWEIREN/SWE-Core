"""浏览器文本转换契约，不调用模型 API。"""

from copy import deepcopy

import pytest

from swe_core.models.utils.content_string import get_content_string


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"content": "你好"}, "你好"),
        ({"content": None}, ""),
        (
            {
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"command": "cat hello.txt"}'
                        }
                    }
                ],
            },
            "```\ncat hello.txt\n```",
        ),
        (
            {"content": '{"returncode": 0, "output": "你好"}'},
            "<returncode>\n0\n<output>\n你好",
        ),
        (
            {"tool_calls": [{"function": {"arguments": "{broken"}}]},
            "```\n{broken\n```",
        ),
        (
            {
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "tool_use", "input": {"command": "pwd"}},
                    {"type": "tool_result", "content": "/workspace"},
                ]
            },
            "inspect\n\n```\npwd\n```\n\n/workspace",
        ),
        (
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "text", "text": "checking"}],
                    },
                    {
                        "type": "function_call",
                        "arguments": '{"command": "ls"}',
                    },
                ]
            },
            "checking\n\n```\nls\n```",
        ),
    ],
)
def test_formats_message_without_mutation(message, expected):
    original = deepcopy(message)
    assert get_content_string(message) == expected
    assert message == original
