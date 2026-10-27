"""将消息内容、工具调用和 observation 转换成可显示文本。"""

import json
from typing import Any


def _format_tool_call(args_str: Any) -> str:
    """格式化工具参数，优先提取 bash command。"""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        if isinstance(args, dict) and "command" in args:
            return f"```\n{args['command']}\n```"
    except Exception:
        pass
    return f"```\n{args_str}\n```"


def _format_observation(content: str) -> str:
    """把包含 returncode 的 JSON observation 展开成键值文本。"""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "returncode" in data:
            lines = []
            for key, value in data.items():
                lines.append(f"<{key}>")
                lines.append(str(value))
            return "\n".join(lines)
        return content
    except Exception:
        return content


def get_content_string(message: dict[str, Any]) -> str:
    """提取普通文本、内容块、工具调用和 Responses output。

    只用于显示历史记录，不执行工具，也不修改消息。
    支持读取某种消息格式不代表 Agent 已接入对应模型接口。
    """
    texts = []

    # 提取字符串或列表形式的 content。
    content = message.get("content")
    if isinstance(content, str):
        texts.append(_format_observation(content))
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_use":
                texts.append(
                    _format_tool_call(json.dumps(item.get("input", {})))
                )
            elif item.get("type") == "tool_result":
                rc = item.get("content", "")
                if isinstance(rc, str):
                    texts.append(_format_observation(rc))
            elif text := item.get("text"):
                texts.append(text)

    # content 为 None 时，仍显示 OpenAI/LiteLLM tool_calls。
    if tool_calls := message.get("tool_calls"):
        for tc in tool_calls:
            func = (
                tc.get("function", {})
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            if func:
                args = (
                    func.get("arguments", "{}")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", "{}")
                )
                texts.append(_format_tool_call(args))

    # 读取 Responses API 格式的历史轨迹。
    if output := message.get("output"):
        if isinstance(output, str):
            texts.append(_format_observation(output))
        elif isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and (text := c.get("text")):
                            texts.append(text)
                elif item.get("type") == "function_call":
                    texts.append(
                        _format_tool_call(item.get("arguments", "{}"))
                    )

    return "\n\n".join(t for t in texts if t)
