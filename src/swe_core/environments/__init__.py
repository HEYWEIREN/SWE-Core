"""Environment 组件工厂。"""

import copy
import importlib

from swe_core import Environment


_ENVIRONMENT_CLASS_MAPPING = {
    "local":
        "swe_core.environments.local.LocalEnvironment",
    "docker":
        "swe_core.environments.docker.DockerEnvironment",
}


def get_environment_class(
    spec: str,
) -> type[Environment]:
    """根据短名称或完整导入路径获取 Environment 类。"""
    full_path = _ENVIRONMENT_CLASS_MAPPING.get(
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
            f"Unknown environment type: {spec} "
            f"(resolved to {full_path}, "
            f"available: {_ENVIRONMENT_CLASS_MAPPING})"
        ) from None


def get_environment(
    config: dict | None = None,
    *,
    default_type: str = "local",
) -> Environment:
    """根据配置创建 Environment 实例。"""
    environment_config = copy.deepcopy(
        config if config is not None else {}
    )

    environment_type = environment_config.pop(
        "environment_class",
        default_type,
    )
    environment_class = get_environment_class(
        environment_type
    )

    return environment_class(
        **environment_config
    )


__all__ = [
    "get_environment",
    "get_environment_class",
]
