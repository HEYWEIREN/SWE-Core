"""配置文件加载工具。"""

import json
import os
from pathlib import Path
from typing import Any

import yaml


builtin_config_dir = Path(__file__).parent

Config = dict[str, Any]


def get_config_path(config_spec: str | Path) -> Path:
    """解析显式文件路径或包内置的 YAML 配置名称。"""
    config_path = Path(config_spec)

    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")

    candidates = [
        config_path,
        Path(
            os.getenv(
                "SWE_CORE_CONFIG_DIR",
                ".",
            )
        )
        / config_path,
        builtin_config_dir / config_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find config file for "
        f"{config_path} (tried: {candidates})"
    )


def _key_value_spec_to_nested_dict(
    config_spec: str,
) -> Config:
    """将点分隔的 ``key=value`` 配置转换为嵌套字典。"""
    key, separator, raw_value = config_spec.partition("=")

    if not separator:
        raise ValueError(
            f"Invalid config spec {config_spec!r}: "
            "expected key=value"
        )

    keys = key.split(".")

    if any(not part for part in keys):
        raise ValueError(
            f"Invalid config spec {config_spec!r}: "
            "empty config key"
        )

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

    result: Config = {}
    current = result

    for part in keys[:-1]:
        child: Config = {}
        current[part] = child
        current = child

    current[keys[-1]] = value
    return result


def get_config_from_spec(
    config_spec: str | Path,
) -> Config:
    """加载一个 YAML 文件，或解析一条 ``key=value`` 配置。"""
    if (
        isinstance(config_spec, str)
        and "=" in config_spec
    ):
        return _key_value_spec_to_nested_dict(
            config_spec
        )

    path = get_config_path(config_spec)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: "
            f"{path}"
        )

    return data


__all__ = [
    "builtin_config_dir",
    "get_config_path",
    "get_config_from_spec",
    "_key_value_spec_to_nested_dict",
]
