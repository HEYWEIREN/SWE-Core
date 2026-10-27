"""Agent 组件工厂。"""

import copy
import importlib

from swe_core import Agent, Environment, Model


_AGENT_CLASS_MAPPING = {
    "default": (
        "swe_core.agents.default.DefaultAgent"
    ),
    "interactive": (
        "swe_core.agents.interactive."
        "InteractiveAgent"
    ),
}


def get_agent_class(spec: str) -> type[Agent]:
    """根据短名称或完整导入路径获取 Agent 类。"""
    full_path = _AGENT_CLASS_MAPPING.get(
        spec,
        spec,
    )

    try:
        module_name, class_name = full_path.rsplit(
            ".",
            1,
        )
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (
        ValueError,
        ImportError,
        AttributeError,
    ):
        raise ValueError(
            f"Unknown agent type: {spec} "
            f"(resolved to {full_path}, "
            f"available: {_AGENT_CLASS_MAPPING})"
        ) from None


def get_agent(
    model: Model,
    env: Environment,
    config: dict | None = None,
    *,
    default_type: str = "default",
) -> Agent:
    """根据配置创建 Agent 实例。"""
    agent_config = copy.deepcopy(
        config if config is not None else {}
    )

    agent_type = agent_config.pop(
        "agent_class",
        default_type,
    )
    agent_class = get_agent_class(agent_type)

    return agent_class(
        model,
        env,
        **agent_config,
    )


__all__ = [
    "get_agent",
    "get_agent_class",
]
