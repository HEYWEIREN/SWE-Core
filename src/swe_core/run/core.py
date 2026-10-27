#!/usr/bin/env python3

"""运行本地 SWE-Core。"""

import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from swe_core.agents import get_agent
from swe_core.agents.utils.prompt_user import (
    multiline_prompt,
)
from swe_core.config import (
    builtin_config_dir,
    get_config_from_spec,
)
from swe_core.environments import (
    get_environment,
)
from swe_core.models import get_model
from swe_core.paths import (
    get_default_trajectory_path,
)
from swe_core.utils.serialize import (
    UNSET,
    recursive_merge,
)


DEFAULT_CONFIG_FILE_ENV = (
    "SWE_CORE_DEFAULT_CONFIG_PATH"
)
MODEL_NAME_ENV = "SWE_CORE_MODEL"


_HELP_TEXT = """
Run SWE-Core in the local environment.

[not dim]
The agent executes shell commands directly on this computer.
Use confirmation mode unless the working directory is trusted.
[/not dim]
"""


_CONFIG_SPEC_HELP_TEXT = """
Path to YAML config files, built-in config names, or key-value pairs.

[bold red]IMPORTANT:[/bold red]
If this option is provided, the default core.yaml is not loaded
automatically. Include it explicitly when you want to inherit its
prompts:

[bold green]
-c core.yaml -c model.model_kwargs.temperature=0
[/bold green]

Multiple config specifications are merged from left to right.
"""


console = Console(highlight=False)

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def get_default_config_file() -> Path:
    """返回 CLI 默认配置文件。"""
    configured_path = os.getenv(
        DEFAULT_CONFIG_FILE_ENV
    )

    if configured_path:
        return Path(configured_path).expanduser()

    return builtin_config_dir / "core.yaml"


def _stdin_is_interactive() -> bool:
    """判断 CLI 是否可以安全读取用户输入。"""
    try:
        return (
            sys.stdin is not None
            and sys.stdin.isatty()
        )
    except (ValueError, OSError):
        return False


def _get_config_section(
    config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """读取并验证一个顶层配置区域。"""
    section = config.get(name, {})

    if not isinstance(section, dict):
        raise ValueError(
            f"Config section {name!r} "
            "must be a mapping"
        )

    return section


def _load_config(
    config_specs: list[str] | None,
    *,
    model_name: str | None,
    model_class: str | None,
    agent_class: str | None,
    environment_class: str | None,
    task: str | None,
    yolo: bool,
    cost_limit: float | None,
    output: Path | None,
    exit_immediately: bool,
) -> tuple[dict[str, Any], list[str]]:
    """加载配置，并在最后应用显式 CLI 参数。"""
    actual_specs = (
        config_specs
        if config_specs
        else [str(get_default_config_file())]
    )

    loaded_configs = [
        get_config_from_spec(spec)
        for spec in actual_specs
    ]

    # 在合并 CLI override 前验证类型，避免错误的列表、字符串
    # 被后续空 mapping 覆盖，导致真正的配置错误被隐藏。
    for loaded_config in loaded_configs:
        for section_name in (
            "run",
            "agent",
            "model",
            "environment",
        ):
            if section_name in loaded_config:
                _get_config_section(
                    loaded_config,
                    section_name,
                )

    cli_overrides = {
        "run": {
            "task": (
                task
                if task
                else UNSET
            ),
        },
        "agent": {
            "agent_class": (
                agent_class
                if agent_class
                else UNSET
            ),
            "mode": (
                "yolo"
                if yolo
                else UNSET
            ),
            "cost_limit": (
                cost_limit
                if cost_limit is not None
                else UNSET
            ),
            "confirm_exit": (
                False
                if exit_immediately
                else UNSET
            ),
            # output=None 表示用户没有显式传入，
            # 不覆盖 YAML 中的 output_path。
            "output_path": (
                output
                if output is not None
                else UNSET
            ),
        },
        "model": {
            "model_class": (
                model_class
                if model_class
                else UNSET
            ),
            "model_name": (
                model_name
                if model_name
                else UNSET
            ),
        },
        "environment": {
            "environment_class": (
                environment_class
                if environment_class
                else UNSET
            ),
        },
    }

    config = recursive_merge(
        *loaded_configs,
        cli_overrides,
    )

    return config, actual_specs


def _resolve_task(
    run_config: dict[str, Any],
) -> str:
    """从配置获取任务，必要时启动多行输入。"""
    task = run_config.get("task", UNSET)

    if task is not UNSET:
        if not isinstance(task, str):
            raise ValueError(
                "run.task must be a string"
            )

        if task.strip():
            return task

    if not _stdin_is_interactive():
        raise typer.BadParameter(
            "No task was provided and stdin is not "
            "interactive. Use --task/-t.",
            param_hint="--task",
        )

    console.print(
        "[bold yellow]"
        "What do you want to do?"
        "[/bold yellow]"
    )

    task = multiline_prompt()

    if not task.strip():
        raise typer.BadParameter(
            "Task cannot be empty.",
            param_hint="--task",
        )

    console.print(
        "[bold green]"
        "Got that, starting the agent."
        "[/bold green]"
    )

    return task


@app.command(help=_HELP_TEXT)
def main(
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model name passed to LiteLLM.",
    ),
    task: str | None = typer.Option(
        None,
        "-t",
        "--task",
        help="Task or problem statement.",
        show_default=False,
    ),
    config_spec: list[str] | None = typer.Option(
        None,
        "-c",
        "--config",
        help=_CONFIG_SPEC_HELP_TEXT,
        show_default=False,
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help=(
            "Output trajectory path. "
            "Defaults to the user config directory."
        ),
        show_default=False,
    ),
    cost_limit: float | None = typer.Option(
        None,
        "-l",
        "--cost-limit",
        help="Cost limit. Set to 0 to disable.",
    ),
    yolo: bool = typer.Option(
        False,
        "-y",
        "--yolo",
        help="Run model commands without confirmation.",
    ),
    exit_immediately: bool = typer.Option(
        False,
        "--exit-immediately",
        help=(
            "Exit immediately when the agent submits "
            "instead of asking for confirmation."
        ),
        rich_help_panel="Advanced",
    ),
    model_class: str | None = typer.Option(
        None,
        "--model-class",
        help=(
            "Model class short name or full "
            "Python import path."
        ),
        rich_help_panel="Advanced",
    ),
    agent_class: str | None = typer.Option(
        None,
        "--agent-class",
        help=(
            "Agent class short name or full "
            "Python import path."
        ),
        rich_help_panel="Advanced",
    ),
    environment_class: str | None = typer.Option(
        None,
        "--environment-class",
        help=(
            "Environment class short name or full "
            "Python import path."
        ),
        rich_help_panel="Advanced",
    ),
) -> Any:
    """装配并运行 Agent。"""
    try:
        config, actual_specs = _load_config(
            config_spec,
            model_name=model_name,
            model_class=model_class,
            agent_class=agent_class,
            environment_class=environment_class,
            task=task,
            yolo=yolo,
            cost_limit=cost_limit,
            output=output,
            exit_immediately=exit_immediately,
        )

        console.print(
            "Building agent from config specs: "
            f"[bold green]{actual_specs}"
            "[/bold green]"
        )

        run_config = _get_config_section(
            config,
            "run",
        )
        agent_config = _get_config_section(
            config,
            "agent",
        )
        model_config = _get_config_section(
            config,
            "model",
        )
        environment_config = _get_config_section(
            config,
            "environment",
        )

        # YAML 和 -m 都没有指定模型时，允许使用
        # Stage 11 smoke test 约定的环境变量。
        if not model_config.get("model_name"):
            environment_model_name = os.getenv(
                MODEL_NAME_ENV
            )

            if environment_model_name:
                model_config["model_name"] = (
                    environment_model_name
                )

        # 仅当 YAML 和 CLI 都没有指定 output_path
        # 时才使用用户目录中的默认路径。
        if "output_path" not in agent_config:
            agent_config["output_path"] = (
                get_default_trajectory_path()
            )

        run_task = _resolve_task(run_config)

        model = get_model(model_config)
        environment = get_environment(
            environment_config,
            default_type="local",
        )
        agent = get_agent(
            model,
            environment,
            agent_config,
            default_type="interactive",
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        console.print(
            "[bold red]Configuration error:[/bold red] "
            f"{error}"
        )
        raise typer.Exit(code=2) from error

    result = agent.run(run_task)

    exit_status = result.get(
        "exit_status",
        "unknown",
    )

    console.print(
        "Agent finished with status "
        f"[bold green]{exit_status}"
        "[/bold green]."
    )

    output_path = agent_config.get(
        "output_path"
    )

    if output_path:
        console.print(
            "Saved trajectory to "
            f"[bold green]'{output_path}'"
            "[/bold green]"
        )

    return agent


if __name__ == "__main__":
    app()
