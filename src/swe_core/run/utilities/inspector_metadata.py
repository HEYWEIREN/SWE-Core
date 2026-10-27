"""将 trajectory 中已经记录的元数据转换为浏览器文本。"""

import math
from datetime import datetime, timezone
from typing import Any


MISSING = "未记录"


def as_mapping(value: Any) -> dict[str, Any]:
    """兼容旧轨迹中缺失或非 mapping 的元数据。"""
    return value if isinstance(value, dict) else {}


def _value(value: Any) -> str:
    """保留零值，避免将缺失数据误显示为零。"""
    if value is None or value == "":
        return MISSING
    return str(value).replace("\x00", "").replace("\n", "\\n")


def _cost(value: Any) -> str:
    """显示已记录的美元成本，保留小额调用的有效数字。"""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return f"${value:.8g}"
    return MISSING


def _limit(value: Any, unit: str = "") -> str:
    """Agent 的零值限额表示禁用，缺失值表示未记录。"""
    if value is None or value == "":
        return MISSING
    if value == 0:
        return "禁用"
    return f"{_value(value)}{unit}"


def _timestamp(value: Any) -> str:
    """将 Unix 时间戳显示为 UTC，不把时间戳解释为执行耗时。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(
                value,
                tz=timezone.utc,
            ).isoformat(timespec="milliseconds")
        except (ValueError, OverflowError, OSError):
            return MISSING
    return _value(value)


def _seconds(value: Any) -> str:
    """为已记录的秒数添加单位，不为未知值添加单位。"""
    return MISSING if value is None or value == "" else f"{_value(value)} s"


def format_trajectory_metadata(
    trajectory: dict[str, Any],
) -> str:
    """显示运行概览，仅读取选定字段，不展开凭据或完整配置。"""
    info = as_mapping(trajectory.get("info"))
    stats = as_mapping(info.get("model_stats"))
    config = as_mapping(info.get("config"))
    model = as_mapping(config.get("model"))
    agent = as_mapping(config.get("agent"))
    environment = as_mapping(config.get("environment"))

    lines = [
        "轨迹概览（m 显示／隐藏元数据）",
        (
            f"退出状态: {_value(info.get('exit_status'))}  |  "
            f"记录总成本: {_cost(stats.get('instance_cost'))}  |  "
            f"模型查询次数: {_value(stats.get('api_calls'))}"
        ),
        (
            f"模型: {_value(model.get('model_name'))}  |  "
            f"环境: {_value(config.get('environment_type'))}"
        ),
        (
            f"Agent: {_value(config.get('agent_type'))}  |  "
            f"交互模式: {_value(agent.get('mode'))}"
        ),
        (
            f"步骤上限: {_limit(agent.get('step_limit'))}  |  "
            f"成本上限: {_limit(agent.get('cost_limit'), ' USD')}  |  "
            f"总时限（配置）: "
            f"{_limit(agent.get('wall_time_limit_seconds'), ' s')}"
        ),
        (
            f"命令超时（配置）: {_seconds(environment.get('timeout'))}"
            f"  |  工作目录: {_value(environment.get('cwd'))}"
        ),
        (
            f"软件版本: {_value(info.get('core_version', info.get('mini_version')))}  |  "
            f"轨迹格式: {_value(trajectory.get('trajectory_format'))}"
        ),
    ]

    if environment.get("image") is not None:
        lines.append(
            f"镜像: {_value(environment['image'])}  |  "
            f"容器存活时长（配置）: "
            f"{_value(environment.get('container_timeout'))}  |  "
            f"容器启动超时（配置）: "
            f"{_seconds(environment.get('pull_timeout'))}"
        )

    if model.get("cost_tracking") == "ignore_errors":
        lines.append(
            "成本提示: ignore_errors 模式下，0 可能表示成本计算失败，"
            "不代表免费。"
        )

    return "\n".join(lines)


def format_message_metadata(
    message: dict[str, Any],
) -> str:
    """展示单条消息的成本、时间、工具状态及已有 provider usage。"""
    extra = as_mapping(message.get("extra"))
    fields = [
        f"消息时间（UTC）: {_timestamp(extra.get('timestamp'))}",
    ]

    if message.get("role") == "assistant" or "cost" in extra:
        fields.append(f"调用成本（记录）: {_cost(extra.get('cost'))}")

    for key, label in (
        ("returncode", "返回码"),
        ("exception_type", "异常类型"),
        ("exception_info", "执行异常"),
        ("exception", "异常详情"),
        ("exception_str", "退出异常"),
        ("interrupt_type", "中断类型"),
        ("exit_status", "退出状态"),
    ):
        if key in extra:
            value = extra[key]
            if key.startswith("exception") and value == "":
                fields.append(f"{label}: 无")
            else:
                fields.append(f"{label}: {_value(value)}")

    if "tool_call_id" in message:
        fields.append(f"工具调用 ID: {_value(message['tool_call_id'])}")

    response = as_mapping(extra.get("response"))
    usage = as_mapping(response.get("usage"))
    if usage:
        # 两种常见的 provider token 字段名称，缺失值不推算。
        for key, alternative, label in (
            ("prompt_tokens", "input_tokens", "输入 tokens"),
            ("completion_tokens", "output_tokens", "输出 tokens"),
            ("total_tokens", "total_tokens", "总 tokens"),
        ):
            value = usage.get(key, usage.get(alternative))
            fields.append(f"{label}: {_value(value)}")

        for key, alternative, token_key, label in (
            (
                "prompt_tokens_details",
                "input_tokens_details",
                "cached_tokens",
                "缓存输入 tokens",
            ),
            (
                "completion_tokens_details",
                "output_tokens_details",
                "reasoning_tokens",
                "推理 tokens",
            ),
        ):
            details = as_mapping(usage.get(key, usage.get(alternative)))
            if token_key in details:
                fields.append(f"{label}: {_value(details[token_key])}")

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = as_mapping(choices[0])
        if "finish_reason" in first_choice:
            fields.append(
                f"finish_reason: {_value(first_choice['finish_reason'])}"
            )

    return "  |  ".join(fields)
