# tests/environments/test_local.py

import shlex
import sys

import pytest

from swe_core.environments.local import LocalEnvironment
from swe_core.exceptions import Submitted


def python_command(source: str) -> str:
    """
    生成使用当前 Python 解释器执行代码的 shell 命令。

    shlex.quote 适用于项目当前的 WSL/Linux bash 环境。
    """
    return (
        f"{shlex.quote(sys.executable)} "
        f"-c {shlex.quote(source)}"
    )


def test_execute_successful_command() -> None:
    environment = LocalEnvironment()

    output = environment.execute(
        {"command": "echo hello"}
    )

    assert output["output"].strip() == "hello"
    assert output["returncode"] == 0
    assert output["exception_info"] == ""


def test_execute_preserves_nonzero_returncode() -> None:
    environment = LocalEnvironment()

    output = environment.execute(
        {
            "command": python_command(
                "import sys; sys.exit(3)"
            )
        }
    )

    assert output["returncode"] == 3
    assert output["exception_info"] == ""


def test_stdout_and_stderr_are_merged() -> None:
    environment = LocalEnvironment()

    output = environment.execute(
        {
            "command": python_command(
                """
import sys
print("normal output")
print("error output", file=sys.stderr)
"""
            )
        }
    )

    assert output["returncode"] == 0
    assert "normal output" in output["output"]
    assert "error output" in output["output"]


def test_timeout_returns_standard_error_output() -> None:
    environment = LocalEnvironment(timeout=1)

    output = environment.execute(
        {
            "command": python_command(
                "import time; time.sleep(5)"
            )
        }
    )

    assert output["returncode"] == -1
    assert output["extra"]["exception_type"] == "TimeoutExpired"
    assert output["exception_info"]


def test_submit_marker_raises_submitted() -> None:
    environment = LocalEnvironment()

    with pytest.raises(Submitted) as exc_info:
        environment.execute(
            {
                "command": (
                    "echo "
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                )
            }
        )

    error = exc_info.value

    assert len(error.messages) == 1

    exit_message = error.messages[0]

    assert exit_message["role"] == "exit"
    assert exit_message["content"] == ""
    assert exit_message["extra"]["exit_status"] == "Submitted"
    assert exit_message["extra"]["submission"] == ""


def test_text_after_submit_marker_becomes_submission() -> None:
    environment = LocalEnvironment()

    with pytest.raises(Submitted) as exc_info:
        environment.execute(
            {
                "command": (
                    "printf "
                    "'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                    "\\nfinal answer\\n'"
                )
            }
        )

    exit_message = exc_info.value.messages[0]

    assert exit_message["role"] == "exit"
    assert exit_message["content"] == "final answer\n"
    assert (
        exit_message["extra"]["submission"]
        == "final answer\n"
    )


def test_submit_marker_must_be_first_output_line() -> None:
    environment = LocalEnvironment()

    output = environment.execute(
        {
            "command": (
                "printf "
                "'some output"
                "\\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                "\\n'"
            )
        }
    )

    assert output["returncode"] == 0
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in output["output"]


def test_submit_requires_zero_returncode() -> None:
    environment = LocalEnvironment()

    output = environment.execute(
        {
            "command": (
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; "
                "exit 2"
            )
        }
    )

    assert output["returncode"] == 2


def test_template_vars_recursively_merge_kwargs() -> None:
    environment = LocalEnvironment(env={"BASE": "1"})

    template_vars = environment.get_template_vars(
        env={"EXTRA": "2"},
    )

    assert template_vars["env"] == {
        "BASE": "1",
        "EXTRA": "2",
    }
