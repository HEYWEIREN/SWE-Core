"""Model 组件工厂。"""

import copy
import importlib

from swe_core import Model


_MODEL_CLASS_MAPPING = {
    "litellm":
        "swe_core.models.litellm_model.LitellmModel",
    "deterministic":
        "swe_core.models.test_models.DeterministicModel",
    "deterministic_toolcall": (
        "swe_core.models.test_models."
        "DeterministicToolcallModel"
    ),
}


def get_model_class(spec: str) -> type[Model]:
    """根据短名称或完整导入路径获取 Model 类。"""
    full_path = _MODEL_CLASS_MAPPING.get(
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
            f"Unknown model type: {spec} "
            f"(resolved to {full_path}, "
            f"available: {_MODEL_CLASS_MAPPING})"
        ) from None


def get_model(
    config: dict | None = None,
    *,
    default_type: str = "litellm",
) -> Model:
    """根据配置创建 Model 实例。"""
    model_config = copy.deepcopy(
        config if config is not None else {}
    )

    model_type = model_config.pop(
        "model_class",
        default_type,
    )
    model_class = get_model_class(model_type)

    return model_class(**model_config)


__all__ = [
    "get_model",
    "get_model_class",
]
