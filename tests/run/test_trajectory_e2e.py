"""CLI 到本地命令执行及 trajectory 的端到端测试。"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from swe_core.models.litellm_model import LitellmModel
from swe_core.models.test_models import make_output
from swe_core.run.core import app


def write_config(
    path: Path,
    config: dict[str, Any],
) -> Path:
    """写入一个 CLI 端到端测试配置。"""
    path.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def load_trajectory(path: Path) -> dict[str, Any]:
    """读取并解析 UTF-8 trajectory。"""
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def assert_trajectory_contract(
    trajectory: dict[str, Any],
    expected_status: str,
) -> None:
    """验证所有退出路径共享的 trajectory 契约。"""
    assert trajectory["trajectory_format"] == (
        "SWE-Core-1.1"
    )
    assert trajectory["info"]["exit_status"] == (
        expected_status
    )

    config = trajectory["info"]["config"]

    assert "agent" in config
    assert "agent_type" in config
    assert "model" in config
    assert "model_type" in config
    assert "environment" in config
    assert "environment_type" in config

    last_message = trajectory["messages"][-1]

    assert last_message["role"] == "exit"
    assert (
        last_message["extra"]["exit_status"]
        == expected_status
    )
    assert "submission" in last_message["extra"]


def make_invalid_litellm_response() -> MagicMock:
    """构造没有 tool call 的 provider response。"""
    response = MagicMock()
    choice = MagicMock()

    choice.finish_reason = "stop"
    choice.message.tool_calls = None
    choice.message.model_dump.return_value = {
        "role": "assistant",
        "content": "invalid response",
    }

    response.choices = [choice]
    response.model_dump.return_value = {
        "id": "invalid-response",
    }

    return response


def test_cli_runs_real_local_environment_and_saves_trajectory(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "run.traj.json"
    config_path = write_config(
        tmp_path / "e2e.yaml",
        {
            "run": {
                "task": "run integration test"
            },
            "agent": {
                "system_template": "system",
                "instance_template": (
                    "Task: {{ task }}"
                ),
                "mode": "confirm",
                "confirm_exit": True,
            },
            "model": {
                "model_class": "deterministic",
                "outputs": [
                    make_output(
                        "inspect",
                        [
                            {
                                "command": (
                                    "printf "
                                    "'integration-ok\\n'"
                                )
                            }
                        ],
                        cost=0.25,
                    ),
                    make_output(
                        "finish",
                        [
                            {
                                "command": (
                                    "printf "
                                    "'COMPLETE_TASK_AND_"
                                    "SUBMIT_FINAL_OUTPUT"
                                    "\\nfinished\\n'"
                                )
                            }
                        ],
                        cost=0.25,
                    ),
                ],
            },
            "environment": {
                "environment_class": "local",
                "cwd": str(tmp_path),
            },
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "--output",
            str(trajectory_path),
            "--yolo",
            "--exit-immediately",
        ],
    )

    assert result.exit_code == 0, result.output
    assert trajectory_path.is_file()
    assert "Submitted" in result.output

    trajectory = load_trajectory(trajectory_path)

    assert_trajectory_contract(
        trajectory,
        "Submitted",
    )
    assert trajectory["info"]["submission"] == (
        "finished\n"
    )
    assert trajectory["info"]["model_stats"] == {
        "instance_cost": 0.5,
        "api_calls": 2,
    }
    agent_config = trajectory["info"]["config"][
        "agent"
    ]
    assert agent_config["mode"] == "yolo"
    assert agent_config["confirm_exit"] is False
    assert any(
        "integration-ok" in message.get("content", "")
        for message in trajectory["messages"]
    )


def test_cli_saves_limits_exceeded_trajectory(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "limit.traj.json"
    config_path = write_config(
        tmp_path / "limit.yaml",
        {
            "run": {
                "task": "reach the step limit",
            },
            "agent": {
                "system_template": "system",
                "instance_template": (
                    "Task: {{ task }}"
                ),
                "step_limit": 1,
                "cost_limit": 0.0,
                "mode": "confirm",
                "confirm_exit": True,
            },
            "model": {
                "model_class": "deterministic",
                "outputs": [
                    make_output(
                        "one step",
                        [
                            {
                                "command": (
                                    "printf 'limit-ok\\n'"
                                )
                            }
                        ],
                        cost=0.25,
                    ),
                ],
            },
            "environment": {
                "environment_class": "local",
                "cwd": str(tmp_path),
            },
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "--output",
            str(trajectory_path),
            "--yolo",
            "--exit-immediately",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "LimitsExceeded" in result.output

    trajectory = load_trajectory(trajectory_path)

    assert_trajectory_contract(
        trajectory,
        "LimitsExceeded",
    )
    assert trajectory["info"]["model_stats"] == {
        "instance_cost": 0.25,
        "api_calls": 1,
    }
    assert any(
        "limit-ok" in message.get("content", "")
        for message in trajectory["messages"]
    )


def test_cli_saves_repeated_format_error_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_path = tmp_path / "format-error.traj.json"
    config_path = write_config(
        tmp_path / "format-error.yaml",
        {
            "run": {
                "task": "produce invalid tool calls",
            },
            "agent": {
                "system_template": "system",
                "instance_template": (
                    "Task: {{ task }}"
                ),
                "max_consecutive_format_errors": 2,
                "cost_limit": 0.0,
                "mode": "confirm",
                "confirm_exit": True,
            },
            "model": {
                "model_class": "litellm",
                "model_name": "openai/test-model",
            },
            "environment": {
                "environment_class": "local",
                "cwd": str(tmp_path),
            },
        },
    )

    response = make_invalid_litellm_response()
    mock_completion = MagicMock(
        return_value=response
    )

    monkeypatch.setattr(
        "swe_core.models.litellm_model."
        "litellm.completion",
        mock_completion,
    )
    monkeypatch.setattr(
        "swe_core.models.litellm_model."
        "litellm.cost_calculator.completion_cost",
        MagicMock(return_value=0.02),
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "--output",
            str(trajectory_path),
            "--yolo",
            "--exit-immediately",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RepeatedFormatError" in result.output
    assert mock_completion.call_count == 2

    trajectory = load_trajectory(trajectory_path)

    assert_trajectory_contract(
        trajectory,
        "RepeatedFormatError",
    )
    assert trajectory["info"]["model_stats"] == {
        "instance_cost": pytest.approx(0.04),
        "api_calls": 2,
    }

    format_errors = [
        message
        for message in trajectory["messages"]
        if message.get("extra", {}).get(
            "interrupt_type"
        ) == "FormatError"
    ]

    assert len(format_errors) == 2
    assert all(
        message["extra"]["response"]
        == {"id": "invalid-response"}
        for message in format_errors
    )


def test_cli_saves_uncaught_exception_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_path = tmp_path / "uncaught.traj.json"
    config_path = write_config(
        tmp_path / "uncaught.yaml",
        {
            "run": {
                "task": "trigger an unexpected error",
            },
            "agent": {
                "system_template": "system",
                "instance_template": (
                    "Task: {{ task }}"
                ),
                "cost_limit": 0.0,
                "mode": "confirm",
                "confirm_exit": True,
            },
            "model": {
                "model_class": "litellm",
                "model_name": "openai/test-model",
            },
            "environment": {
                "environment_class": "local",
                "cwd": str(tmp_path),
            },
        },
    )

    monkeypatch.setattr(
        "swe_core.models.litellm_model."
        "litellm.completion",
        MagicMock(return_value=MagicMock()),
    )

    def raise_cost_error(
        self: LitellmModel,
        response: object,
    ) -> dict[str, float]:
        raise RuntimeError(
            "trajectory e2e failure"
        )

    monkeypatch.setattr(
        LitellmModel,
        "_calculate_cost",
        raise_cost_error,
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "--output",
            str(trajectory_path),
            "--yolo",
            "--exit-immediately",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(
        result.exception,
        RuntimeError,
    )
    assert trajectory_path.is_file()

    trajectory = load_trajectory(trajectory_path)

    assert_trajectory_contract(
        trajectory,
        "RuntimeError",
    )

    last_message = trajectory["messages"][-1]

    assert last_message["extra"]["exception_str"] == (
        "trajectory e2e failure"
    )
    assert "RuntimeError: trajectory e2e failure" in (
        last_message["extra"]["traceback"]
    )
