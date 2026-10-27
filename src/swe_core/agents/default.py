"""Default agent implementation."""

import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from swe_core import Environment, Model, __version__
from swe_core.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    TimeExceeded,
)
from swe_core.utils.serialize import recursive_merge


TRAJECTORY_FORMAT = "SWE-Core-1.1"


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """
    将文本原子地写入目标文件。

    先在目标文件所在目录创建临时文件，完整写入并同步，
    再通过 os.replace() 原子替换目标文件。

    如果写入或替换失败：
    - 原目标文件保持不变；
    - 临时文件会被清理；
    - 原始异常继续向调用者抛出。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )


class AgentConfig(BaseModel):
    """
    DefaultAgent 的最小配置。
    """

    system_template: str
    instance_template: str

    step_limit: int = 0
    cost_limit: float = 3.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3

    output_path: Path | None = None


class DefaultAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type[AgentConfig] = AgentConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)

        self.model = model
        self.env = env

        self.messages: list[dict[str, Any]] = []
        self.extra_template_vars: dict[str, Any] = {}

        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0

        self._start_time = time.time()

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        汇总 Agent、Model、Environment 的模板变量。

        后面的字典优先级更高，因此运行时 kwargs
        可以覆盖默认配置。
        """
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(
                    time.time() - self._start_time
                ),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(
        self,
        template: str,
    ) -> str:
        """
        使用严格模式渲染 Jinja2 模板。

        缺少变量时直接报错，不允许静默生成空字符串。
        """
        return Template(
            template,
            undefined=StrictUndefined,
        ).render(
            **self.get_template_vars()
        )

    def add_messages(
        self,
        *messages: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        将消息追加到完整运行轨迹中。
        """
        self.messages.extend(messages)
        return list(messages)

    def run(
        self,
        task: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        初始化消息，并不断调用 step()，直到出现 exit 消息。
        """
        self.extra_template_vars = recursive_merge(
            self.extra_template_vars,
            {
                "task": task,
                **kwargs,
            },
        )

        self.messages = []

        system_message = self.model.format_message(
            role="system",
            content=self._render_template(
                self.config.system_template
            ),
        )

        instance_message = self.model.format_message(
            role="user",
            content=self._render_template(
                self.config.instance_template
            ),
        )

        self.add_messages(
            system_message,
            instance_message,
        )

        while True:
            try:
                self.step()

                # 只要某一步完整执行成功，
                # 连续格式错误计数就清零。
                self.n_consecutive_format_errors = 0

            except FormatError as error:
                self.n_consecutive_format_errors += 1

                # 某些真实模型在解析失败前已经产生费用。
                if error.messages:
                    self.cost += (
                        error.messages[0]
                        .get("extra", {})
                        .get("cost", 0.0)
                    )

                if (
                    0
                    < self.config.max_consecutive_format_errors
                    <= self.n_consecutive_format_errors
                ):
                    self.add_messages(
                        *error.messages,
                        self.model.format_message(
                            role="exit",
                            content="RepeatedFormatError",
                            extra={
                                "exit_status":
                                    "RepeatedFormatError",
                                "submission": "",
                            },
                        ),
                    )
                else:
                    self.add_messages(*error.messages)

            except InterruptAgentFlow as error:
                # Submitted、LimitsExceeded、TimeExceeded
                # 最终都会进入这里。
                self.add_messages(*error.messages)

            except Exception as error:
                self.handle_uncaught_exception(error)
                raise

            finally:
                # 无论成功、提交、超限还是异常，
                # 都保存当前轨迹。
                self.save(self.config.output_path)

            if (
                self.messages
                and self.messages[-1].get("role") == "exit"
            ):
                break

        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict[str, Any]]:
        """
        完成一次 model -> environment -> observation。
        """
        message = self.query()
        return self.execute_actions(message)

    def query(self) -> dict[str, Any]:
        """
        检查限制，调用模型，并保存 assistant 消息。
        """
        if (
            0 < self.config.step_limit <= self.n_calls
            or
            0 < self.config.cost_limit <= self.cost
        ):
            raise LimitsExceeded(
                self.model.format_message(
                    role="exit",
                    content="LimitsExceeded",
                    extra={
                        "exit_status": "LimitsExceeded",
                        "submission": "",
                    },
                )
            )

        elapsed_seconds = int(
            time.time() - self._start_time
        )

        if (
            0
            < self.config.wall_time_limit_seconds
            <= elapsed_seconds
        ):
            raise TimeExceeded(
                self.model.format_message(
                    role="exit",
                    content="TimeExceeded",
                    extra={
                        "exit_status": "TimeExceeded",
                        "submission": "",
                    },
                )
            )

        self.n_calls += 1

        message = self.model.query(self.messages)

        self.cost += (
            message
            .get("extra", {})
            .get("cost", 0.0)
        )

        self.add_messages(message)

        return message

    def execute_actions(
        self,
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        执行 assistant message 中的全部 actions，
        再由 Model 将输出包装成 observation。
        """
        actions = (
            message
            .get("extra", {})
            .get("actions", [])
        )

        outputs = [
            self.env.execute(action)
            for action in actions
        ]

        observation_messages = (
            self.model.format_observation_messages(
                message,
                outputs,
                self.get_template_vars(),
            )
        )

        return self.add_messages(
            *observation_messages
        )

    def handle_uncaught_exception(
        self,
        error: Exception,
    ) -> list[dict[str, Any]]:
        """
        将未预期异常记录成 exit 消息。

        run() 随后仍会重新抛出异常，
        但 finally 会先保存 trajectory。
        """
        exit_message = self.model.format_message(
            role="exit",
            content=str(error),
            extra={
                "exit_status":
                    type(error).__name__,
                "submission": "",
                "exception_str": str(error),
                "traceback": traceback.format_exc(),
            },
        )

        return self.add_messages(exit_message)

    def serialize(
        self,
        *extra_dicts: dict[str, Any],
    ) -> dict[str, Any]:
        """
        将 Agent、Model、Environment 状态组合成
        可以写入 JSON 的字典。
        """
        last_message = (
            self.messages[-1]
            if self.messages
            else {}
        )

        last_extra = last_message.get(
            "extra",
            {},
        )

        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent":
                        self.config.model_dump(
                            mode="json"
                        ),
                    "agent_type": (
                        f"{self.__class__.__module__}."
                        f"{self.__class__.__name__}"
                    ),
                },
                "core_version": __version__,
                "exit_status":
                    last_extra.get(
                        "exit_status",
                        "",
                    ),
                "submission":
                    last_extra.get(
                        "submission",
                        "",
                    ),
            },
            "messages": self.messages,
            "trajectory_format": TRAJECTORY_FORMAT,
        }

        return recursive_merge(
            agent_data,
            self.model.serialize(),
            self.env.serialize(),
            *extra_dicts,
        )

    def save(
        self,
        path: Path | None,
        *extra_dicts: dict[str, Any],
    ) -> dict[str, Any]:
        """
        保存 trajectory。

        path 为 None 时只返回序列化结果，不写文件。

        有目标路径时先完成 JSON 序列化，再原子替换目标文件。
        序列化或写入失败不会破坏上一份有效 trajectory。
        """
        data = self.serialize(*extra_dicts)

        if path is not None:
            path = Path(path)

            # 必须先完成序列化，再创建和写入临时文件。
            # 如果数据不能序列化，旧 trajectory 不受影响。
            content = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

            _atomic_write_text(
                path,
                content,
            )

        return data
