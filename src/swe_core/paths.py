"""用户级配置与运行状态路径。"""

import os
from pathlib import Path

from platformdirs import user_config_dir


APP_NAME = "SWE-Core"
GLOBAL_CONFIG_DIR_ENV = (
    "SWE_CORE_GLOBAL_CONFIG_DIR"
)


def get_global_config_dir() -> Path:
    """返回用户级配置和运行状态目录。

    设置 SWE_CORE_GLOBAL_CONFIG_DIR 时优先使用
    该目录，否则使用 platformdirs 提供的平台默认目录。

    本函数只计算路径，不创建目录。
    """
    configured_path = os.getenv(
        GLOBAL_CONFIG_DIR_ENV
    )

    if configured_path:
        return Path(
            configured_path
        ).expanduser()

    return Path(
        user_config_dir(APP_NAME)
    )


def ensure_global_config_dir() -> Path:
    """创建并返回用户级配置目录。"""
    directory = get_global_config_dir()
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    return directory


def get_default_trajectory_path() -> Path:
    """返回默认 trajectory 文件路径。"""
    return (
        get_global_config_dir()
        / "last_core_run.traj.json"
    )


def get_interactive_history_path() -> Path:
    """返回交互式输入历史文件路径。"""
    return (
        get_global_config_dir()
        / "interactive_history.txt"
    )


__all__ = [
    "APP_NAME",
    "GLOBAL_CONFIG_DIR_ENV",
    "ensure_global_config_dir",
    "get_default_trajectory_path",
    "get_global_config_dir",
    "get_interactive_history_path",
]