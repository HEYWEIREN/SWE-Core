# src/swe_core/environments/local.py

import os
import platform
import signal
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from swe_core.exceptions import Submitted
from swe_core.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel):
    """LocalEnvironment 的配置。"""

    cwd: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    timeout: int = 30


class LocalEnvironment:
    """直接在本地机器上执行 shell 命令。"""

    def __init__(
        self,
        *,
        config_class: type[LocalEnvironmentConfig] = LocalEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        执行一条 shell 命令，并把结果转换为统一的 observation dict。
        """

        command = action.get("command", "")

        # 优先级：
        # 本次调用传入的 cwd
        # -> 环境配置中的 cwd
        # -> 当前 Python 进程所在目录
        cwd = cwd or self.config.cwd or os.getcwd()

        try:
            result = _run(
                command,
                cwd,
                os.environ | self.config.env,
                timeout or self.config.timeout,
            )

            output: dict[str, Any] = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }

        except Exception as error:
            # TimeoutExpired 等异常可能携带已经产生的输出。
            raw_output = getattr(error, "output", None)

            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode(
                    "utf-8",
                    errors="replace",
                )
            else:
                raw_output = raw_output or ""

            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": (
                    "An error occurred while executing the command: "
                    f"{error}"
                ),
                "extra": {
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                },
            }

        # 必须放在 except Exception 的外面。
        self._check_finished(output)

        return output

    def _check_finished(self, output: dict[str, Any]) -> None:
        """
        检查命令输出是否表示任务已经完成。

        第一行必须为：
        COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

        并且 returncode 必须为 0。
        """

        raw_output = output.get("output", "")

        lines = raw_output.lstrip().splitlines(keepends=True)

        # lstrip()：去掉字符串开头的空格、\n、\t 等空白
        # splitlines()：按行拆成列表
        # keepends=True：拆分后保留每行末尾的换行符

        if (
            lines
            and lines[0].strip()
            == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            # 标记行后面的内容作为最终提交内容。
            submission = "".join(lines[1:])

            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                    },
                }
            )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """
        返回后续渲染 prompt 时可以使用的环境变量。
        """

        return recursive_merge(
            self.config.model_dump(),
            platform.uname()._asdict(),
            dict(os.environ),
            kwargs,
        )

    def serialize(self) -> dict[str, Any]:
        """返回可写入 trajectory 的环境配置。"""

        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": (
                        f"{self.__class__.__module__}."
                        f"{self.__class__.__name__}"
                    ),
                }
            }
        }


def _run(
    command: str,
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """
    启动 shell 执行命令。

    与简单的 subprocess.run 相比，这里在超时时会终止整个进程组，
    避免命令创建的子进程继续留在后台。
    """

    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )

    try:
        stdout, _ = process.communicate(timeout=timeout)

    except subprocess.TimeoutExpired:
        if os.name == "posix":
            # WSL/Linux 下杀死整个进程组。
            os.killpg(process.pid, signal.SIGKILL)
        else:
            # Windows 下至少终止当前 shell 进程。
            process.kill()

        # 回收进程，并拿到超时前已经输出的内容。
        stdout, _ = process.communicate()

        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
        )

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
    )
