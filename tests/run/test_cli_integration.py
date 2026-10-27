"""真实命令入口的 subprocess 集成测试。"""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import pytest
import yaml

from swe_core.models.test_models import (
    make_toolcall_output,
)


ENTRYPOINTS = [
    "python-module",
    "core",
    "SWE-Core",
]


def get_entrypoint_command(
    entrypoint: str,
) -> list[str]:
    """返回指定 CLI 入口对应的命令前缀。"""
    if entrypoint == "python-module":
        return [
            sys.executable,
            "-m",
            "swe_core",
        ]

    executable_suffix = (
        ".exe"
        if os.name == "nt"
        else ""
    )

    scripts_directory = Path(
        sysconfig.get_path("scripts")
    )
    executable_path = (
        scripts_directory
        / f"{entrypoint}{executable_suffix}"
    )

    if not executable_path.is_file():
        pytest.fail(
            f"Console script does not exist: "
            f"{executable_path}. "
            "Install the project with "
            "`.venv/bin/python -m pip install -e .`."
        )

    return [
        str(executable_path),
    ]


def make_bash_output(
    content: str,
    command: str,
    *,
    tool_call_id: str,
) -> dict[str, Any]:
    """构造 deterministic tool-call 模型输出。"""
    return make_toolcall_output(
        content,
        [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps(
                        {
                            "command": command,
                        }
                    ),
                },
            }
        ],
        [
            {
                "command": command,
                "tool_call_id": tool_call_id,
            }
        ],
    )


def write_subprocess_config(
    path: Path,
    *,
    workdir: Path,
) -> Path:
    """创建真实 CLI subprocess 使用的配置。"""
    create_command = (
        "printf 'hello\\n' > hello.txt "
        "&& cat hello.txt"
    )

    submit_command = (
        'test "$(cat hello.txt)" = "hello" '
        "&& printf "
        "'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        "\\nsubprocess complete\\n'"
    )

    config = {
        "run": {
            "task": (
                "Create hello.txt, verify its content, "
                "and submit the result."
            ),
        },
        "agent": {
            "system_template": (
                "You are a coding agent."
            ),
            "instance_template": (
                "Task: {{ task }}"
            ),
            "step_limit": 3,
            "cost_limit": 0.0,
            "mode": "confirm",
            "confirm_exit": True,
        },
        "model": {
            "model_class": (
                "deterministic_toolcall"
            ),
            "outputs": [
                make_bash_output(
                    "Create and inspect hello.txt.",
                    create_command,
                    tool_call_id="call-create",
                ),
                make_bash_output(
                    "Verify and submit.",
                    submit_command,
                    tool_call_id="call-submit",
                ),
            ],
        },
        "environment": {
            "environment_class": "local",
            "cwd": str(workdir),
            "timeout": 10,
        },
    }

    path.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return path


def assert_successful_process(
    result: subprocess.CompletedProcess[str],
) -> None:
    """提供包含 stdout/stderr 的 subprocess 失败信息。"""
    assert result.returncode == 0, (
        f"Command failed with exit code "
        f"{result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "entrypoint",
    ENTRYPOINTS,
)
def test_cli_entrypoint_help(
    entrypoint: str,
    tmp_path: Path,
) -> None:
    """验证三个真实 CLI 入口均可显示帮助。"""
    command = [
        *get_entrypoint_command(entrypoint),
        "--help",
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert_successful_process(result)

    assert "--task" in result.stdout
    assert "--config" in result.stdout
    assert "--model" in result.stdout
    assert "--output" in result.stdout
    assert "--yolo" in result.stdout


@pytest.mark.parametrize(
    "entrypoint",
    ENTRYPOINTS,
)
def test_cli_entrypoint_runs_local_environment_and_saves_trajectory(
    entrypoint: str,
    tmp_path: Path,
) -> None:
    """
    验证三个真实命令入口均可完成：

    CLI
      -> DeterministicToolcallModel
      -> LocalEnvironment
      -> Submitted
      -> trajectory
    """
    config_path = write_subprocess_config(
        tmp_path / "subprocess.yaml",
        workdir=tmp_path,
    )
    trajectory_path = (
        tmp_path
        / f"{entrypoint}.traj.json"
    )

    command = [
        *get_entrypoint_command(entrypoint),
        "--config",
        str(config_path),
        "--output",
        str(trajectory_path),
        "--yolo",
        "--exit-immediately",
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert_successful_process(result)

    assert "Submitted" in result.stdout
    assert "Saved trajectory to" in result.stdout

    hello_path = tmp_path / "hello.txt"

    assert hello_path.is_file()
    assert (
        hello_path.read_text(encoding="utf-8")
        == "hello\n"
    )

    assert trajectory_path.is_file()

    trajectory = json.loads(
        trajectory_path.read_text(
            encoding="utf-8"
        )
    )

    assert trajectory["trajectory_format"] == (
        "SWE-Core-1.1"
    )
    assert trajectory["info"]["exit_status"] == (
        "Submitted"
    )
    assert trajectory["info"]["submission"] == (
        "subprocess complete\n"
    )
    assert trajectory["info"]["model_stats"] == {
        "instance_cost": 2.0,
        "api_calls": 2,
    }

    config = trajectory["info"]["config"]

    assert config["agent_type"].endswith(
        ".InteractiveAgent"
    )
    assert config["model_type"].endswith(
        ".DeterministicToolcallModel"
    )
    assert config["environment_type"].endswith(
        ".LocalEnvironment"
    )

    agent_config = config["agent"]

    assert agent_config["mode"] == "yolo"
    assert agent_config["confirm_exit"] is False
    assert agent_config["output_path"] == str(
        trajectory_path
    )

    roles = [
        message["role"]
        for message in trajectory["messages"]
    ]

    for required_role in (
        "system",
        "user",
        "assistant",
        "tool",
        "exit",
    ):
        assert required_role in roles

    assert (
        trajectory["messages"][-1]["role"]
        == "exit"
    )
    assert (
        trajectory["messages"][-1]["extra"]
        ["exit_status"]
        == "Submitted"
    )

    assert any(
        "hello" in message.get("content", "")
        for message in trajectory["messages"]
        if message.get("role") == "tool"
    )
