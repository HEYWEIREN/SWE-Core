"""真实 CLI 产物与 Textual 浏览器之间的集成验收。"""

import json
import subprocess
import sys
import sysconfig
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from textual.widgets import Static

from swe_core.run.utilities.inspector import TrajectoryInspector


def screen_text(app: TrajectoryInspector) -> str:
    """读取实际挂载到界面的文本。"""
    return "\n".join(
        str(widget.content)
        for widget in app.query("#content Static")
        if isinstance(widget, Static) and widget.display
    )


@pytest.mark.parametrize(
    "entrypoint",
    ["module", "core-extra", "core-e"],
)
def test_installed_inspector_help(entrypoint, tmp_path):
    if entrypoint == "module":
        command = [
            sys.executable,
            "-m",
            "swe_core.run.utilities.inspector",
        ]
    else:
        command = [
            str(Path(sysconfig.get_path("scripts")) / entrypoint),
            "inspect",
        ]

    result = subprocess.run(
        [*command, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--reasoning" in result.stdout


async def test_real_cli_trajectory_is_browsable(tmp_path):
    """真实命令执行、文件保存、命令与结果显示及重载全部连通。"""
    commands = [
        "printf 'hello-inspector\n' > hello.txt && cat hello.txt",
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n'",
    ]
    outputs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            ],
            "extra": {
                "actions": [
                    {"command": command, "tool_call_id": f"call-{index}"}
                ],
                "cost": 0.0,
            },
        }
        for index, command in enumerate(commands)
    ]
    config = {
        "model": {
            "model_class": "deterministic_toolcall",
            "outputs": outputs,
        },
        "environment": {"cwd": str(tmp_path)},
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    trajectory_path = tmp_path / "actual.traj.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swe_core",
            "-c",
            "core.yaml",
            "-c",
            str(config_path),
            "-t",
            "Create a file and submit.",
            "-o",
            str(trajectory_path),
            "-y",
            "--exit-immediately",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "hello.txt").read_text() == "hello-inspector\n"
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["info"]["exit_status"] == "Submitted"

    app = TrajectoryInspector([trajectory_path])
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "SYSTEM" in screen_text(app)
        assert "退出状态: Submitted" in screen_text(app)
        assert "记录总成本: $0" in screen_text(app)
        assert "模型查询次数: 2" in screen_text(app)
        assert "LocalEnvironment" in screen_text(app)
        await pilot.press("l")
        await pilot.pause()
        assert "cat hello.txt" in screen_text(app)
        assert "hello-inspector" in screen_text(app)
        assert "TOOL" in screen_text(app)
        assert "调用成本（记录）: $0" in screen_text(app)
        assert "返回码: 0" in screen_text(app)
        await pilot.press("$")
        await pilot.pause()
        assert "EXIT" in screen_text(app)
        assert "done" in screen_text(app)
        await pilot.press("q")


@pytest.mark.parametrize(
    "data",
    [{"messages": None}, {"messages": [42]}, {"messages": "bad"}],
)
def test_malformed_messages_are_reported(tmp_path, data):
    path = tmp_path / "bad.traj.json"
    path.write_text(json.dumps(data))
    with patch.object(TrajectoryInspector, "notify") as notify:
        app = TrajectoryInspector([path])
    assert app.messages == []
    assert app.steps == []
    assert notify.call_args.kwargs["severity"] == "error"


@pytest.mark.parametrize(
    "format_name",
    ["SWE-Core-1.0", "SWE-Core-1.1", "compatible-agent-1.1"],
)
def test_existing_format_names_remain_readable(tmp_path, format_name):
    path = tmp_path / "compatible.traj.json"
    messages = [{"role": "user", "content": "旧轨迹"}]
    path.write_text(
        json.dumps({"messages": messages, "trajectory_format": format_name}),
        encoding="utf-8",
    )
    assert TrajectoryInspector([path]).messages == messages


async def test_jless_missing_and_failed_launch_cleanup(tmp_path):
    path = tmp_path / "jless.traj.json"
    path.write_text('[{"role": "user", "content": "hello"}]')
    app = TrajectoryInspector([path])
    async with app.run_test():
        with (
            patch.object(app, "suspend"),
            patch.object(app, "notify") as notify,
            patch(
                "swe_core.run.utilities.inspector.subprocess.run",
                side_effect=FileNotFoundError,
            ),
        ):
            app._open_in_jless(path)
            assert "jless not found" in notify.call_args.args[0]

        paths = []

        def fail_open(temp_path):
            paths.append(temp_path)
            assert temp_path.exists()
            raise RuntimeError("viewer failed")

        with patch.object(app, "_open_in_jless", side_effect=fail_open):
            with pytest.raises(RuntimeError, match="viewer failed"):
                app.action_open_in_jless()

        assert len(paths) == 1
        assert not paths[0].exists()
