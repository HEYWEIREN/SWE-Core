"""通过 Docker CLI 执行 Agent 命令。"""

import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from swe_core.exceptions import Submitted
from swe_core.utils.serialize import recursive_merge


class DockerEnvironmentConfig(BaseModel):
    """DockerEnvironment 的配置。"""

    image: str

    cwd: str = "/"
    """容器内执行命令的工作目录。"""

    env: dict[str, str] = {}
    """显式传给容器内命令的环境变量。"""

    forward_env: list[str] = []
    """从宿主机转发的环境变量名称。

    只转发宿主机已设置的变量；发生冲突时，env 优先。
    """

    timeout: int = 30
    """等待 docker exec 完成的超时时间，单位为秒。"""

    executable: str = os.getenv(
        "SWE_CORE_DOCKER_EXECUTABLE",
        "docker",
    )
    """Docker 或兼容容器命令的可执行文件路径。"""

    run_args: list[str] = ["--rm"]
    """docker run 的附加参数，默认在容器退出后自动删除容器。"""

    container_timeout: str = "2h"
    """容器保持运行的时长，格式与 sleep 命令一致。"""

    pull_timeout: int = 120
    """启动容器的超时时间，包含可能发生的镜像拉取。"""

    interpreter: list[str] = ["bash", "-lc"]
    """执行命令的解释器及参数，实际命令会追加在末尾。

    可以移除 -l 以禁用 login shell，或指定其他解释器。
    """


class DockerEnvironment:
    """通过 Docker CLI 在持续运行的容器内执行命令。"""

    def __init__(
        self,
        *,
        config_class: type[DockerEnvironmentConfig] = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        self.logger = logger or logging.getLogger(
            "swe_core.environment"
        )
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """合并环境配置、宿主机平台信息和额外模板变量。"""
        return recursive_merge(
            self.config.model_dump(),
            platform.uname()._asdict(),
            kwargs,
        )

    def serialize(self) -> dict[str, Any]:
        """返回可写入 trajectory 的环境配置。"""
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(
                        mode="json"
                    ),
                    "environment_type": (
                        f"{self.__class__.__module__}."
                        f"{self.__class__.__name__}"
                    ),
                }
            }
        }

    def _start_container(self) -> None:
        """启动容器，并保存 Docker 返回的容器 ID。"""
        container_name = f"swe_core-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(
            f"Starting container with command: {shlex.join(cmd)}"
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # 首次运行可能需要拉取镜像。
            timeout=self.config.pull_timeout,
            check=True,
        )
        self.logger.info(
            f"Started container {container_name} "
            f"with ID {result.stdout.strip()}"
        )
        self.container_id = result.stdout.strip()

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """执行容器内命令，并返回标准化 observation。"""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [
            self.config.executable,
            "exec",
            "-w",
            cwd,
        ]

        # 先转发宿主机变量，再追加显式配置，确保 env 优先。
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend(
            [
                self.container_id,
                *self.config.interpreter,
                command,
            ]
        )

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }

        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": (
                    f"An error occurred while executing the command: {e}"
                ),
                "extra": {
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                },
            }

        # Submitted 是控制流异常，必须在 except Exception 外抛出。
        self._check_finished(output)
        return output

    def _check_finished(
        self,
        output: dict[str, Any],
    ) -> None:
        """成功命令的首条非空输出行为提交标记时，抛出 Submitted。"""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip()
            == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
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

    def cleanup(self) -> None:
        """后台停止容器，失败时尝试强制删除。"""
        # 初始化可能提前失败，container_id 属性不一定存在。
        if getattr(self, "container_id", None) is not None:
            cmd = (
                f"(timeout 60 {self.config.executable} "
                f"stop {self.container_id} || "
                f"{self.config.executable} rm -f {self.container_id}) "
                ">/dev/null 2>&1 &"
            )
            subprocess.Popen(cmd, shell=True)

    def __del__(self) -> None:
        """对象销毁时触发容器清理。"""
        self.cleanup()
