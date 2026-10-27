from pathlib import Path

import pytest

from swe_core.paths import (
    ensure_global_config_dir,
    get_default_trajectory_path,
    get_global_config_dir,
    get_interactive_history_path,
)


def test_environment_overrides_global_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    monkeypatch.setenv(
        "SWE_CORE_GLOBAL_CONFIG_DIR",
        str(directory),
    )

    assert get_global_config_dir() == directory
    assert get_default_trajectory_path() == (
        directory / "last_core_run.traj.json"
    )
    assert get_interactive_history_path() == (
        directory / "interactive_history.txt"
    )
    assert not directory.exists()


def test_ensure_global_directory_creates_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    monkeypatch.setenv(
        "SWE_CORE_GLOBAL_CONFIG_DIR",
        str(directory),
    )

    assert ensure_global_config_dir() == directory
    assert directory.is_dir()
