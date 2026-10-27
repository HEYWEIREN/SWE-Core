"""由环境变量显式启用的真实模型 CLI smoke test。"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


LIVE_TEST_ENABLED = (
    os.getenv("RUN_LIVE_MODEL_TEST") == "1"
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_TEST_ENABLED,
        reason=(
            "Set RUN_LIVE_MODEL_TEST=1 to call "
            "a real model API."
        ),
    ),
]


def load_trajectory(
    path: Path,
) -> dict[str, Any]:
    """读取真实模型运行生成的 trajectory。"""
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_live_model_creates_verifies_and_submits_file(
    tmp_path: Path,
) -> None:
    """通过真实 CLI 和模型完成最小本地编码任务。"""
    model_name = os.getenv(
        "SWE_CORE_MODEL"
    )

    if not model_name:
        pytest.fail(
            "SWE_CORE_MODEL must be set when "
            "RUN_LIVE_MODEL_TEST=1."
        )

    trajectory_path = (
        tmp_path / "live-model.traj.json"
    )
    task = (
        "In the current working directory, create "
        "hello.txt with exactly the text hello followed "
        "by a newline. Run cat hello.txt to verify the "
        "content before submitting. Do not modify any "
        "other file."
    )

    command = [
        sys.executable,
        "-m",
        "swe_core",
        "--config",
        "core.yaml",
        "--config",
        (
            "environment.cwd="
            f"{json.dumps(str(tmp_path))}"
        ),
        "--config",
        "agent.step_limit=6",
        "--config",
        "agent.wall_time_limit_seconds=180",
        "--model",
        model_name,
        "--task",
        task,
        "--output",
        str(trajectory_path),
        "--cost-limit",
        "1.0",
        "--yolo",
        "--exit-immediately",
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f"Live CLI failed with exit code "
        f"{result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    hello_path = tmp_path / "hello.txt"

    assert hello_path.is_file()
    assert (
        hello_path.read_text(encoding="utf-8")
        == "hello\n"
    )
    assert trajectory_path.is_file()

    trajectory = load_trajectory(
        trajectory_path
    )

    assert trajectory["trajectory_format"] == (
        "SWE-Core-1.1"
    )
    assert trajectory["info"]["exit_status"] == (
        "Submitted"
    )

    model_stats = trajectory["info"][
        "model_stats"
    ]

    assert 2 <= model_stats["api_calls"] <= 6
    assert model_stats["instance_cost"] >= 0.0

    config = trajectory["info"]["config"]

    assert config["model"]["model_name"] == (
        model_name
    )
    assert config["model_type"].endswith(
        ".LitellmModel"
    )
    assert config["environment_type"].endswith(
        ".LocalEnvironment"
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

    actions = [
        action
        for message in trajectory["messages"]
        for action in (
            message.get("extra", {}).get(
                "actions",
                [],
            )
        )
    ]

    assert any(
        "hello.txt" in action.get("command", "")
        for action in actions
    )
    assert any(
        "cat hello.txt" in action.get("command", "")
        for action in actions
    )
    assert any(
        "hello" in message.get("content", "")
        for message in trajectory["messages"]
        if message.get("role") == "tool"
    )

    last_message = trajectory["messages"][-1]

    assert last_message["role"] == "exit"
    assert (
        last_message["extra"]["exit_status"]
        == "Submitted"
    )
