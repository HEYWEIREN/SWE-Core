import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import StrictUndefined, Template

from swe_core.agents.default import AgentConfig
from swe_core.agents.interactive import (
    InteractiveAgentConfig,
)
from swe_core.environments.local import LocalEnvironmentConfig
from swe_core.models.litellm_model import LitellmModelConfig


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "swe_core"
    / "config"
    / "core.yaml"
)


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def render(template: str, **context: Any) -> str:
    return Template(
        template,
        undefined=StrictUndefined,
    ).render(**context)


def test_yaml_loads_and_component_configs_accept_it(
    config: dict[str, Any],
) -> None:
    agent = AgentConfig(**config["agent"])
    interactive_agent = InteractiveAgentConfig(
        **config["agent"]
    )
    environment = LocalEnvironmentConfig(**config["environment"])
    model = LitellmModelConfig(
        model_name="test/model",
        **config["model"],
    )

    assert agent.system_template
    assert interactive_agent.mode == "confirm"
    assert interactive_agent.whitelist_actions == []
    assert interactive_agent.confirm_exit is True
    assert environment.env["PAGER"] == "cat"
    assert model.model_kwargs["drop_params"] is True


def test_agent_prompts_render_required_protocol(
    config: dict[str, Any],
) -> None:
    system_prompt = render(config["agent"]["system_template"])
    instance_prompt = render(
        config["agent"]["instance_template"],
        task="Fix the parser",
        system="Linux",
        release="6.8",
        version="#1",
        machine="x86_64",
    )
    normalized_prompt = instance_prompt.lower()

    assert "helpful assistant" in system_prompt
    assert all(
        text.lower() in normalized_prompt
        for text in (
            "Fix the parser",
            "at least one bash tool call",
            "new subshell",
            "not persistent",
            "Analyze the codebase",
            "reproduce the issue",
            "Verify your fix",
            "Test edge cases",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "Do not combine",
            "cannot continue",
            "Linux 6.8 #1 x86_64",
        )
    )


def test_short_observation_is_valid_json(
    config: dict[str, Any],
) -> None:
    raw_output = 'line 1\n"quoted"\\path'
    raw_output += "x" * (9_999 - len(raw_output))
    output = {
        "returncode": 7,
        "output": raw_output,
        "exception_info": None,
    }

    observation = json.loads(
        render(
            config["model"]["observation_template"],
            output=output,
        )
    )

    assert observation["returncode"] == 7
    assert observation["output"] == output["output"]
    assert "exception_info" not in observation


def test_long_observation_keeps_head_and_tail(
    config: dict[str, Any],
) -> None:
    raw_output = "H" * 5_000 + "omitted" + "T" * 5_000

    observation = json.loads(
        render(
            config["model"]["observation_template"],
            output={
                "returncode": 0,
                "output": raw_output,
                "exception_info": None,
            },
        )
    )

    assert observation == {
        "returncode": 0,
        "output_head": "H" * 5_000,
        "output_tail": "T" * 5_000,
        "elided_chars": len("omitted"),
        "warning": "Output too long.",
    }


def test_exception_observation_returns_exception_info(
    config: dict[str, Any],
) -> None:
    observation = json.loads(
        render(
            config["model"]["observation_template"],
            output={
                "returncode": 124,
                "output": "partial output",
                "exception_info": "timed out\nafter 5s",
            },
        )
    )

    assert observation["returncode"] == 124
    assert observation["exception_info"] == "timed out\nafter 5s"


@pytest.mark.parametrize(
    ("context", "expected", "unexpected"),
    [
        (
            {"error": "invalid arguments", "has_tool_calls": True},
            "invalid arguments",
            "output token limit",
        ),
        (
            {
                "error": "unused",
                "has_tool_calls": False,
                "finish_reason": "length",
            },
            "finish_reason=length",
            "Tool call error",
        ),
    ],
)
def test_format_error_distinguishes_failure_types(
    config: dict[str, Any],
    context: dict[str, Any],
    expected: str,
    unexpected: str,
) -> None:
    message = render(
        config["model"]["format_error_template"],
        **context,
    )

    assert expected in message
    assert unexpected not in message
    assert "bash" in message.lower()
