"""A minimal software engineering agent package."""

__version__ = "0.1.0"


from pathlib import Path
from typing import Any, Protocol


class Model(Protocol):
    """语言模型组件的最小接口。"""

    config: Any

    def query(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """根据当前消息历史生成下一条 assistant 消息."""
        ...

    def format_message(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构造一条符合内部格式的消息."""
        ...

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """把环境执行结果转换成可以继续发送给模型的消息."""
        ...

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """返回模型提供给提示词模板的变量."""
        ...

    def serialize(self) -> dict[str, Any]:
        """返回可写入 trajectory 的模型信息."""
        ...


class Environment(Protocol):
    """命令执行环境的最小接口。"""

    config: Any

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
    ) -> dict[str, Any]:
        """执行一个 action，并返回标准化结果."""
        ...

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """返回环境相关的模板变量."""
        ...

    def serialize(self) -> dict[str, Any]:
        """返回可序列化的环境信息."""
        ...


class Agent(Protocol):
    """Agent 控制器的最小接口。"""

    config: Any

    def run(
        self,
        task: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """运行任务，直到提交、超限或异常退出."""
        ...

    def save(
        self,
        path: Path | None,
        *extra_dicts: dict[str, Any],
    ) -> dict[str, Any]:
        """保存 Agent 当前运行轨迹."""
        ...


__all__ = [
    "Agent",
    "Environment",
    "Model",
]