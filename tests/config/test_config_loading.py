from pathlib import Path

import pytest

from swe_core.config import (
    _key_value_spec_to_nested_dict,
    get_config_from_spec,
    get_config_path,
)
from swe_core.utils.serialize import recursive_merge


def test_loads_builtin_config() -> None:
    path = get_config_path("core")
    config = get_config_from_spec("core")

    assert path.name == "core.yaml"
    assert set(config) == {
        "agent",
        "environment",
        "model",
    }


def test_loads_explicit_and_environment_config_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(
        "agent:\n  cost_limit: 1.5\n",
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    environment_path = config_dir / "custom.yaml"
    environment_path.write_text(
        "environment:\n  timeout: 12\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SWE_CORE_CONFIG_DIR",
        str(config_dir),
    )

    assert get_config_from_spec(explicit_path) == {
        "agent": {"cost_limit": 1.5}
    }
    assert get_config_from_spec("custom") == {
        "environment": {"timeout": 12}
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            "agent.cost_limit=0",
            {"agent": {"cost_limit": 0}},
        ),
        (
            "agent.confirm_exit=false",
            {"agent": {"confirm_exit": False}},
        ),
        (
            "model.model_kwargs.temperature=0.5",
            {
                "model": {
                    "model_kwargs": {
                        "temperature": 0.5,
                    }
                }
            },
        ),
        (
            'environment.env={"DEBUG":"1"}',
            {
                "environment": {
                    "env": {"DEBUG": "1"},
                }
            },
        ),
        (
            "model.model_name=test/model",
            {
                "model": {
                    "model_name": "test/model",
                }
            },
        ),
    ],
)
def test_parses_key_value_types(
    spec: str,
    expected: dict,
) -> None:
    assert (
        _key_value_spec_to_nested_dict(spec)
        == expected
    )


@pytest.mark.parametrize(
    "spec",
    [
        "=value",
        ".agent.mode=confirm",
        "agent..mode=confirm",
        "agent.mode.=confirm",
    ],
)
def test_rejects_empty_config_keys(
    spec: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="empty config key",
    ):
        _key_value_spec_to_nested_dict(spec)


def test_config_specs_merge_in_order() -> None:
    config = recursive_merge(
        get_config_from_spec("core"),
        get_config_from_spec(
            "agent.cost_limit=0"
        ),
        get_config_from_spec(
            "model.model_kwargs.temperature=0.5"
        ),
    )

    assert config["agent"]["cost_limit"] == 0
    assert (
        config["model"]["model_kwargs"]
        ["temperature"]
        == 0.5
    )
    assert config["agent"]["system_template"]


def test_rejects_missing_and_non_mapping_configs(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Could not find config file",
    ):
        get_config_path("does-not-exist")

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(
        "- first\n- second\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="YAML mapping",
    ):
        get_config_from_spec(invalid_path)


def test_empty_yaml_loads_as_empty_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert get_config_from_spec(path) == {}
