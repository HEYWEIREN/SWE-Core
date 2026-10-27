"""显式启用的构建与干净安装验收测试。"""

import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest


PACKAGING_TEST_ENABLED = (
    os.getenv("RUN_PACKAGING_TEST") == "1"
)
PROJECT_ROOT = Path(__file__).parents[2]

pytestmark = [
    pytest.mark.packaging,
    pytest.mark.skipif(
        not PACKAGING_TEST_ENABLED,
        reason=(
            "Set RUN_PACKAGING_TEST=1 to build and "
            "install the package in a clean venv."
        ),
    ),
]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """运行命令，并在失败时展示完整输出。"""
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    assert result.returncode == 0, (
        f"Command failed: {command}\n"
        f"exit code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    return result


def get_venv_python(
    environment_path: Path,
) -> Path:
    """返回临时虚拟环境中的 Python 路径。"""
    if os.name == "nt":
        return (
            environment_path
            / "Scripts"
            / "python.exe"
        )

    return environment_path / "bin" / "python"


def get_venv_script(
    environment_path: Path,
    name: str,
) -> Path:
    """返回临时虚拟环境中的 console-script 路径。"""
    if os.name == "nt":
        return (
            environment_path
            / "Scripts"
            / f"{name}.exe"
        )

    return environment_path / "bin" / name


def test_build_and_install_wheel_in_clean_environment(
    tmp_path: Path,
) -> None:
    """构建 wheel，并在新虚拟环境验证三个入口。"""
    distribution_directory = tmp_path / "dist"

    run_command(
        [
            sys.executable,
            "-m",
            "build",
            "--outdir",
            str(distribution_directory),
            str(PROJECT_ROOT),
        ],
        cwd=PROJECT_ROOT,
    )

    wheels = list(
        distribution_directory.glob("*.whl")
    )
    source_distributions = list(
        distribution_directory.glob("*.tar.gz")
    )

    assert len(wheels) == 1
    assert len(source_distributions) == 1

    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_files = set(wheel.namelist())

    assert "swe_core/__main__.py" in wheel_files
    assert "swe_core/config/inspector.tcss" in wheel_files
    assert (
        "swe_core/run/utilities/inspector.py"
        in wheel_files
    )
    assert (
        "swe_core/run/utilities/inspector_metadata.py"
        in wheel_files
    )
    assert (
        "swe_core/config/core.yaml"
        in wheel_files
    )

    environment_path = tmp_path / "clean-venv"
    venv.EnvBuilder(
        with_pip=True,
        clear=True,
    ).create(environment_path)

    environment_python = get_venv_python(
        environment_path
    )

    run_command(
        [
            str(environment_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel_path),
        ],
        cwd=tmp_path,
    )

    module_help = run_command(
        [
            str(environment_python),
            "-m",
            "swe_core",
            "--help",
        ],
        cwd=tmp_path,
    )

    assert "--task" in module_help.stdout
    assert "--config" in module_help.stdout

    for script_name in (
        "core",
        "SWE-Core",
    ):
        script_path = get_venv_script(
            environment_path,
            script_name,
        )

        assert script_path.is_file()

        script_help = run_command(
            [
                str(script_path),
                "--help",
            ],
            cwd=tmp_path,
        )

        assert "--task" in script_help.stdout
        assert "--output" in script_help.stdout

    for script_name in ("core-extra", "core-e"):
        script_help = run_command(
            [
                str(get_venv_script(environment_path, script_name)),
                "inspect",
                "--help",
            ],
            cwd=tmp_path,
        )
        assert "--reasoning" in script_help.stdout

    # 在仓库外、全新解释器中真正挂载界面，验证 wheel 内的样式资源。
    inspector_smoke = """
import asyncio
import json
from pathlib import Path
from swe_core.run.utilities.inspector import TrajectoryInspector

path = Path("installed.traj.json")
path.write_text(json.dumps({"messages": [
    {"role": "user", "content": "installed wheel"},
    {"role": "assistant", "content": "ready"}
]}), encoding="utf-8")

async def check():
    app = TrajectoryInspector([path])
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.n_steps == 2
        await pilot.press("l")
        assert app.i_step == 1
        assert app.query_one("#content")
        await pilot.press("q")

asyncio.run(check())
print("Installed inspector rendered successfully")
"""
    result = run_command(
        [str(environment_python), "-c", inspector_smoke],
        cwd=tmp_path,
    )
    assert "Installed inspector rendered successfully" in result.stdout
