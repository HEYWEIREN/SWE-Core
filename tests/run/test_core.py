"""CLI 装配与端到端运行测试。"""

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import swe_core.run.core as core


runner = CliRunner()
PROJECT_ROOT = Path(__file__).parents[2]


class StubAgent:
    """记录 CLI 传入的任务，避免聚焦测试调用真实模型。"""

    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(self, task: str) -> dict[str, str]:
        self.tasks.append(task)
        return {
            "exit_status": "Submitted",
            "submission": "done",
        }


def write_yaml(
    path: Path,
    data: dict[str, Any],
) -> Path:
    """将一个测试配置写入临时目录。"""
    path.write_text(
        yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def install_component_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], StubAgent]:
    """替换组件工厂，同时保留 CLI 的装配过程。"""
    captured: dict[str, Any] = {}
    agent = StubAgent()
    model = object()
    environment = object()

    def fake_get_model(
        config: dict[str, Any],
    ) -> object:
        captured["model"] = config
        return model

    def fake_get_environment(
        config: dict[str, Any],
        *,
        default_type: str,
    ) -> object:
        captured["environment"] = config
        captured["environment_default"] = default_type
        return environment

    def fake_get_agent(
        actual_model: object,
        actual_environment: object,
        config: dict[str, Any],
        *,
        default_type: str,
    ) -> StubAgent:
        assert actual_model is model
        assert actual_environment is environment
        captured["agent"] = config
        captured["agent_default"] = default_type
        return agent

    monkeypatch.setattr(core, "get_model", fake_get_model)
    monkeypatch.setattr(
        core,
        "get_environment",
        fake_get_environment,
    )
    monkeypatch.setattr(core, "get_agent", fake_get_agent)

    return captured, agent


def test_help_and_console_scripts_expose_cli() -> None:
    result = runner.invoke(core.app, ["--help"])

    assert result.exit_code == 0
    assert "--task" in result.output
    assert "--config" in result.output
    assert "--yolo" in result.output
    assert "--exit-immediately" in result.output

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    scripts = project["project"]["scripts"]

    assert scripts["core"] == "swe_core.run.core:app"
    assert (
        scripts["SWE-Core"]
        == "swe_core.run.core:app"
    )


def test_cli_merges_specs_then_applies_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_yaml(
        tmp_path / "base.yaml",
        {
            "run": {"task": "task from yaml"},
            "agent": {
                "step_limit": 1,
                "cost_limit": 9.0,
                "mode": "confirm",
                "confirm_exit": True,
            },
            "model": {"model_name": "yaml-model"},
            "environment": {"timeout": 10},
        },
    )
    output_path = tmp_path / "cli.traj.json"
    captured, agent = install_component_stubs(
        monkeypatch
    )

    result = runner.invoke(
        core.app,
        [
            "-c",
            str(config_path),
            "-c",
            "agent.step_limit=7",
            "--task",
            "task from cli",
            "--model",
            "cli-model",
            "--model-class",
            "deterministic",
            "--agent-class",
            "interactive",
            "--environment-class",
            "local",
            "--cost-limit",
            "1.5",
            "--yolo",
            "--exit-immediately",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert agent.tasks == ["task from cli"]
    assert captured["model"] == {
        "model_name": "cli-model",
        "model_class": "deterministic",
    }
    assert captured["environment"] == {
        "timeout": 10,
        "environment_class": "local",
    }
    assert captured["agent"] == {
        "step_limit": 7,
        "cost_limit": 1.5,
        "mode": "yolo",
        "confirm_exit": False,
        "agent_class": "interactive",
        "output_path": output_path,
    }
    assert captured["agent_default"] == "interactive"
    assert captured["environment_default"] == "local"
    assert "Submitted" in result.output
    assert str(output_path) in result.output


def test_tty_without_task_uses_multiline_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_yaml(
        tmp_path / "prompt.yaml",
        {},
    )
    captured, agent = install_component_stubs(
        monkeypatch
    )
    monkeypatch.setattr(
        core,
        "_stdin_is_interactive",
        lambda: True,
    )
    monkeypatch.setattr(
        core,
        "multiline_prompt",
        lambda: "line one\nline two",
    )

    result = runner.invoke(
        core.app,
        ["--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert agent.tasks == ["line one\nline two"]
    assert captured["agent_default"] == "interactive"
    assert "agent_class" not in captured["agent"]
    assert "What do you want to do?" in result.output
    assert "starting the agent" in result.output


def test_non_tty_without_task_returns_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_yaml(
        tmp_path / "no-task.yaml",
        {},
    )
    monkeypatch.setattr(
        core,
        "_stdin_is_interactive",
        lambda: False,
    )

    result = runner.invoke(
        core.app,
        ["--config", str(config_path)],
    )

    assert result.exit_code == 2
    assert "No task was provided" in result.output
    assert "--task" in result.output


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        (
            {
                "run": {"task": "test"},
                "agent": [],
            },
            "must be a mapping",
        ),
        (
            {
                "run": {"task": "test"},
                "model": {
                    "model_class": "unknown-model",
                },
            },
            "Unknown model type",
        ),
    ],
)
def test_configuration_errors_exit_with_code_two(
    tmp_path: Path,
    config: dict[str, Any],
    expected_error: str,
) -> None:
    config_path = write_yaml(
        tmp_path / "invalid.yaml",
        config,
    )

    result = runner.invoke(
        core.app,
        ["--config", str(config_path)],
    )

    assert result.exit_code == 2
    assert "Configuration error" in result.output
    assert expected_error in result.output
