from copy import deepcopy
from typing import Callable

import pytest

from swe_core.agents import (
    get_agent,
    get_agent_class,
)
from swe_core.agents.default import DefaultAgent
from swe_core.agents.interactive import InteractiveAgent
from swe_core.environments import (
    get_environment,
    get_environment_class,
)
from swe_core.environments.local import LocalEnvironment
from swe_core.models import (
    get_model,
    get_model_class,
)
from swe_core.models.litellm_model import LitellmModel
from swe_core.models.test_models import (
    DeterministicModel,
    DeterministicToolcallModel,
)


@pytest.mark.parametrize(
    ("resolver", "spec", "expected"),
    [
        (get_agent_class, "default", DefaultAgent),
        (
            get_agent_class,
            "interactive",
            InteractiveAgent,
        ),
        (
            get_environment_class,
            "local",
            LocalEnvironment,
        ),
        (get_model_class, "litellm", LitellmModel),
        (
            get_model_class,
            "deterministic",
            DeterministicModel,
        ),
        (
            get_model_class,
            "deterministic_toolcall",
            DeterministicToolcallModel,
        ),
    ],
)
def test_resolves_short_class_names(
    resolver: Callable[[str], type],
    spec: str,
    expected: type,
) -> None:
    assert resolver(spec) is expected


@pytest.mark.parametrize(
    ("resolver", "spec", "expected"),
    [
        (
            get_agent_class,
            "swe_core.agents.default.DefaultAgent",
            DefaultAgent,
        ),
        (
            get_environment_class,
            (
                "swe_core.environments.local."
                "LocalEnvironment"
            ),
            LocalEnvironment,
        ),
        (
            get_model_class,
            (
                "swe_core.models.litellm_model."
                "LitellmModel"
            ),
            LitellmModel,
        ),
    ],
)
def test_resolves_full_import_paths(
    resolver: Callable[[str], type],
    spec: str,
    expected: type,
) -> None:
    assert resolver(spec) is expected


def test_factories_create_components() -> None:
    model = get_model(
        {
            "model_class": "deterministic",
            "outputs": [],
        }
    )
    environment = get_environment()
    agent = get_agent(
        model,
        environment,
        {
            "agent_class": "default",
            "system_template": "system",
            "instance_template": "{{ task }}",
        },
    )

    assert isinstance(model, DeterministicModel)
    assert isinstance(environment, LocalEnvironment)
    assert isinstance(agent, DefaultAgent)


def test_factory_creates_interactive_agent() -> None:
    model = get_model(
        {
            "model_class": "deterministic",
            "outputs": [],
        }
    )
    environment = get_environment()

    agent = get_agent(
        model,
        environment,
        {
            "agent_class": "interactive",
            "system_template": "system",
            "instance_template": "{{ task }}",
        },
    )

    assert isinstance(agent, InteractiveAgent)
    assert agent.config.mode == "confirm"


def test_factories_do_not_modify_input_configs() -> None:
    model_config = {
        "model_class": "litellm",
        "model_name": "test/model",
        "model_kwargs": {"temperature": 0},
    }
    environment_config = {
        "environment_class": "local",
        "env": {"DEBUG": "1"},
    }
    agent_config = {
        "agent_class": "default",
        "system_template": "system",
        "instance_template": "{{ task }}",
    }
    original_configs = deepcopy(
        (
            model_config,
            environment_config,
            agent_config,
        )
    )

    model = get_model(model_config)
    environment = get_environment(environment_config)
    get_agent(model, environment, agent_config)

    assert (
        model_config,
        environment_config,
        agent_config,
    ) == original_configs


@pytest.mark.parametrize(
    ("resolver", "spec", "message"),
    [
        (
            get_agent_class,
            "unknown",
            "Unknown agent type",
        ),
        (
            get_environment_class,
            "unknown",
            "Unknown environment type",
        ),
        (
            get_model_class,
            "unknown",
            "Unknown model type",
        ),
    ],
)
def test_unknown_component_has_clear_error(
    resolver: Callable[[str], type],
    spec: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolver(spec)
