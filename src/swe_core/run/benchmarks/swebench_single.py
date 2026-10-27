"""运行单个 SWE-bench 实例，支持按实例 ID 或排序后的索引选择。"""

from pathlib import Path

import typer
from datasets import load_dataset

from swe_core.agents import get_agent
from swe_core.config import builtin_config_dir, get_config_from_spec
from swe_core.models import get_model
from swe_core.paths import get_global_config_dir
from swe_core.run.benchmarks.swebench import (
    DATASET_MAPPING,
    get_sb_environment,
)
from swe_core.utils.log import logger
from swe_core.utils.serialize import UNSET, recursive_merge

DEFAULT_OUTPUT_FILE = (
    get_global_config_dir() / "last_swebench_single_run.traj.json"
)
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c benchmarks/swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c benchmarks/swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c benchmarks/swebench.yaml -c agent.mode=yolo[/bold green]
"""


@app.command()
def main(
    subset: str = typer.Option(
        "lite",
        "--subset",
        help="SWEBench subset to use or path to a dataset",
        rich_help_panel="Data selection",
    ),
    split: str = typer.Option(
        "dev",
        "--split",
        help="Dataset split",
        rich_help_panel="Data selection",
    ),
    instance_spec: str = typer.Option(
        0,
        "-i",
        "--instance",
        help="SWE-Bench instance ID or index",
        rich_help_panel="Data selection",
    ),
    model_name: str | None = typer.Option(
        None, "-m", "--model", help="Model to use", rich_help_panel="Basic"
    ),
    model_class: str | None = typer.Option(
        None,
        "--model-class",
        help="Model class to use (e.g., 'litellm' or 'swe_core.models.litellm_model.LitellmModel')",
        rich_help_panel="Advanced",
    ),
    agent_class: str | None = typer.Option(
        None,
        "--agent-class",
        help="Agent class to use (e.g., 'interactive' or 'swe_core.agents.interactive.InteractiveAgent')",
        rich_help_panel="Advanced",
    ),
    environment_class: str | None = typer.Option(
        None,
        "--environment-class",
        help="Environment class to use (e.g., 'docker' or 'swe_core.environments.docker.DockerEnvironment')",
        rich_help_panel="Advanced",
    ),
    yolo: bool = typer.Option(
        False, "-y", "--yolo", help="Run without confirmation"
    ),
    cost_limit: float | None = typer.Option(
        None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."
    ),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)],
        "-c",
        "--config",
        help=_CONFIG_SPEC_HELP_TEXT,
        rich_help_panel="Basic",
    ),
    exit_immediately: bool = typer.Option(
        False,
        "--exit-immediately",
        help="Exit immediately when the agent wants to finish instead of prompting.",
        rich_help_panel="Advanced",
    ),
    output: Path | None = typer.Option(
        DEFAULT_OUTPUT_FILE,
        "-o",
        "--output",
        help="Output trajectory file",
        rich_help_panel="Basic",
    ),
) -> None:
    """运行单个 SWE-bench 实例，支持按实例 ID 或排序后的索引选择。"""
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_path}, split {split}...")
    instances = {
        inst["instance_id"]: inst  # type: ignore
        for inst in load_dataset(dataset_path, split=split)
    }
    # 数字按排序后的索引选择，其余输入直接作为实例 ID。
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "agent": {
                "agent_class": agent_class or UNSET,
                "mode": "yolo" if yolo else UNSET,
                "cost_limit": cost_limit if cost_limit is not None else UNSET,
                "confirm_exit": False if exit_immediately else UNSET,
                "output_path": output or UNSET,
            },
            "model": {
                "model_class": model_class or UNSET,
                "model_name": model_name or UNSET,
            },
            "environment": {
                "environment_class": environment_class or UNSET,
            },
        }
    )
    config = recursive_merge(*configs)

    env = get_sb_environment(config, instance)
    agent = get_agent(
        get_model(config=config.get("model", {})),
        env,
        config.get("agent", {}),
        default_type="interactive",
    )
    agent.run(instance["problem_statement"])


if __name__ == "__main__":
    app()
