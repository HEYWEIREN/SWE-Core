"""辅助命令入口。

提供轨迹浏览器及 SWE-bench、ProgramBench 评测入口。
"""

import sys
from importlib import import_module

from rich.console import Console


subcommands = [
    (
        "swe_core.run.benchmarks.swebench",
        ["swebench"],
        "Run SWE-bench instances in batch mode",
    ),
    (
        "swe_core.run.benchmarks.swebench_single",
        ["swebench-single"],
        "Run a single SWE-bench instance interactively",
    ),
    (
        "swe_core.run.benchmarks.programbench",
        ["programbench"],
        "Run ProgramBench instances in batch mode",
    ),
    (
        "swe_core.run.utilities.inspector",
        ["inspect", "i", "inspector"],
        "Browse saved trajectories",
    ),
]


def get_docstring() -> str:
    """生成已实现的辅助命令说明。"""
    lines = [
        "Extra commands for SWE-Core.",
        "",
        "Available sub-commands:",
        "",
    ]

    for _, aliases, description in subcommands:
        alias_text = " or ".join(
            f"[bold green]{alias}[/bold green]" for alias in aliases
        )
        lines.append(f"  {alias_text}: {description}")

    return "\n".join(lines)


def main() -> None:
    """按首个参数选择工具，其余参数交给对应 Typer 应用。"""
    args = sys.argv[1:]

    if not args or (len(args) == 1 and args[0] in ("-h", "--help")):
        Console().print(get_docstring())
        return

    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            import_module(module_path).app(
                args[1:],
                prog_name=f"core-extra {aliases[0]}",
            )
            return

    Console().print(get_docstring())


if __name__ == "__main__":
    main()
