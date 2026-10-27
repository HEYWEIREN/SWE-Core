"""已有 trajectory 元数据的格式化与实际界面显示回归测试。"""

import copy
import json

import pytest
from textual.widgets import Static

from swe_core.run.utilities.inspector import TrajectoryInspector
from swe_core.run.utilities.inspector_metadata import (
    format_message_metadata,
    format_trajectory_metadata,
)


@pytest.fixture
def trajectory():
    return {
        "trajectory_format": "SWE-Core-1.1",
        "info": {
            "core_version": "0.1.0",
            "exit_status": "Submitted",
            "model_stats": {"instance_cost": 0.0123, "api_calls": 2},
            "config": {
                "agent_type": "swe_core.agents.interactive.InteractiveAgent",
                "environment_type": "swe_core.environments.docker.DockerEnvironment",
                "agent": {
                    "mode": "yolo",
                    "step_limit": 20,
                    "cost_limit": 3,
                    "wall_time_limit_seconds": 120,
                },
                "model": {
                    "model_name": "test-model",
                    "cost_tracking": "ignore_errors",
                    "model_kwargs": {"api_key": "secret-model-key"},
                },
                "environment": {
                    "image": "python:3.12-slim",
                    "cwd": "/workspace",
                    "timeout": 30,
                    "container_timeout": "2h",
                    "pull_timeout": 180,
                    "env": {"TOKEN": "secret-env-token"},
                },
            },
        },
        "messages": [
            {"role": "user", "content": "test task"},
            {
                "role": "assistant",
                "content": "[bold]literal command[/bold]",
                "reasoning_content": "reasoning text",
                "extra": {
                    "timestamp": 0,
                    "cost": 0.00000123,
                    "response": {
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 8,
                            "total_tokens": 20,
                            "prompt_tokens_details": {"cached_tokens": 0},
                            "completion_tokens_details": {
                                "reasoning_tokens": 3
                            },
                        },
                        "choices": [{"finish_reason": "tool_calls"}],
                    },
                },
            },
            {
                "role": "tool",
                "content": "partial output",
                "tool_call_id": "call-1",
                "extra": {
                    "returncode": -1,
                    "exception_info": "Timed out after 30 seconds.",
                },
            },
        ],
    }


def screen_text(app):
    return "\n".join(
        str(widget.content) for widget in app.query("#content Static")
    )


def test_overview_displays_selected_recorded_fields(trajectory):
    before = copy.deepcopy(trajectory)
    text = format_trajectory_metadata(trajectory)
    for expected in (
        "退出状态: Submitted",
        "记录总成本: $0.0123",
        "模型查询次数: 2",
        "模型: test-model",
        "DockerEnvironment",
        "InteractiveAgent",
        "交互模式: yolo",
        "步骤上限: 20",
        "成本上限: 3 USD",
        "总时限（配置）: 120 s",
        "命令超时（配置）: 30 s",
        "工作目录: /workspace",
        "软件版本: 0.1.0",
        "轨迹格式: SWE-Core-1.1",
        "镜像: python:3.12-slim",
        "容器存活时长（配置）: 2h",
        "容器启动超时（配置）: 180 s",
        "不代表免费",
    ):
        assert expected in text
    assert "secret-model-key" not in text
    assert "secret-env-token" not in text
    assert trajectory == before


@pytest.mark.parametrize(
    "info", [None, [], "bad", {}, {"config": None, "model_stats": []}]
)
def test_missing_overview_fields_are_not_zero(info):
    text = format_trajectory_metadata({"info": info})
    assert "记录总成本: 未记录" in text
    assert "模型查询次数: 未记录" in text
    assert "命令超时（配置）: 未记录" in text
    assert "$0" not in text
    assert "禁用" not in text


def test_zero_stats_and_disabled_limits(trajectory):
    trajectory["info"]["model_stats"] = {"instance_cost": 0, "api_calls": 0}
    agent = trajectory["info"]["config"]["agent"]
    for key in ("step_limit", "cost_limit", "wall_time_limit_seconds"):
        agent[key] = 0
    text = format_trajectory_metadata(trajectory)
    assert "记录总成本: $0" in text
    assert "模型查询次数: 0" in text
    assert "步骤上限: 禁用" in text
    assert "成本上限: 禁用" in text
    assert "总时限（配置）: 禁用" in text


def test_message_time_cost_usage_and_finish_reason(trajectory):
    message = trajectory["messages"][1]
    before = copy.deepcopy(message)
    text = format_message_metadata(message)
    for expected in (
        "消息时间（UTC）: 1970-01-01T00:00:00.000+00:00",
        "调用成本（记录）: $1.23e-06",
        "输入 tokens: 12",
        "输出 tokens: 8",
        "总 tokens: 20",
        "缓存输入 tokens: 0",
        "推理 tokens: 3",
        "finish_reason: tool_calls",
    ):
        assert expected in text
    assert "耗时" not in text
    assert message == before


def test_responses_usage_does_not_invent_total():
    text = format_message_metadata(
        {
            "extra": {
                "response": {
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 9,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 2},
                    }
                }
            },
        }
    )
    assert "输入 tokens: 0" in text
    assert "输出 tokens: 9" in text
    assert "总 tokens: 未记录" in text
    assert "缓存输入 tokens: 0" in text
    assert "推理 tokens: 2" in text


@pytest.mark.parametrize(
    "extra",
    [None, [], "bad", {}, {"response": {"usage": [], "choices": [None]}}],
)
def test_missing_message_metadata(extra):
    text = format_message_metadata({"role": "assistant", "extra": extra})
    assert "消息时间（UTC）: 未记录" in text
    assert "调用成本（记录）: 未记录" in text
    assert "tokens" not in text


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 1e100])
def test_invalid_timestamp_is_unknown(timestamp):
    text = format_message_metadata({"extra": {"timestamp": timestamp}})
    assert text == "消息时间（UTC）: 未记录"


def test_execution_and_interruption_metadata():
    text = format_message_metadata(
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "extra": {
                "returncode": 0,
                "exception_info": "",
                "exception_type": "LimitsExceeded",
                "exception": "wall time exceeded",
                "exception_str": "run interrupted",
                "interrupt_type": "LimitsExceeded",
                "exit_status": "LimitsExceeded",
            },
        }
    )
    for expected in (
        "返回码: 0",
        "执行异常: 无",
        "异常类型: LimitsExceeded",
        "异常详情: wall time exceeded",
        "退出异常: run interrupted",
        "中断类型: LimitsExceeded",
        "退出状态: LimitsExceeded",
        "工具调用 ID: call-2",
    ):
        assert expected in text


async def test_metadata_visible_toggleable_and_plain_text(
    tmp_path, trajectory
):
    trajectory["info"]["config"]["model"]["model_name"] = "[red]model[/red]"
    path = tmp_path / "full.traj.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")
    app = TrajectoryInspector([path])
    async with app.run_test() as pilot:
        await pilot.pause()
        overview = app.query_one(".trajectory-metadata", Static).content
        assert "[red]model[/red]" in str(overview)
        assert not overview.spans
        await pilot.press("l")
        await pilot.pause()
        assert "记录总成本: $0.0123" in screen_text(app)
        assert "总 tokens: 20" in screen_text(app)
        assert "返回码: -1" in screen_text(app)
        assert "Timed out after 30 seconds." in screen_text(app)
        assert "reasoning text" in screen_text(app)
        await pilot.press("m")
        await pilot.pause()
        assert not app.query(".trajectory-metadata, .message-metadata")
        assert app.i_step == 1
        assert "partial output" in screen_text(app)
        assert "reasoning text" in screen_text(app)
        await pilot.press("r", "m")
        await pilot.pause()
        assert "记录总成本: $0.0123" in screen_text(app)
        assert "reasoning text" not in screen_text(app)
        assert app.i_step == 1


async def test_reload_and_file_switch_do_not_reuse_stale_metadata(
    tmp_path, trajectory
):
    full = tmp_path / "full.traj.json"
    full.write_text(json.dumps(trajectory), encoding="utf-8")
    legacy = tmp_path / "legacy.traj.json"
    legacy.write_text(
        '[{"role": "user", "content": "old"}]', encoding="utf-8"
    )
    app = TrajectoryInspector([full, legacy])
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "记录总成本: $0.0123" in screen_text(app)
        trajectory["info"]["model_stats"]["instance_cost"] = 0.5
        full.write_text(json.dumps(trajectory), encoding="utf-8")
        await pilot.press("R")
        await pilot.pause()
        assert "记录总成本: $0.5" in screen_text(app)
        await pilot.press("L")
        await pilot.pause()
        assert "记录总成本: 未记录" in screen_text(app)
        assert "test-model" not in screen_text(app)
        await pilot.press("H")
        await pilot.pause()
        assert "记录总成本: $0.5" in screen_text(app)
        full.write_text("invalid json", encoding="utf-8")
        await pilot.press("R")
        await pilot.pause()
        assert "记录总成本: 未记录" in screen_text(app)
        assert "test-model" not in screen_text(app)
        assert app.trajectory == {}


async def test_empty_messages_still_display_overview(tmp_path, trajectory):
    trajectory["messages"] = []
    path = tmp_path / "empty.traj.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")
    app = TrajectoryInspector([path])
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "记录总成本: $0.0123" in screen_text(app)
        assert "退出状态: Submitted" in screen_text(app)
