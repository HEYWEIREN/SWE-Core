import logging
import time
from typing import Any

from pydantic import BaseModel

from swe_core.models.utils.actions_text import format_observation_messages
from swe_core.models.utils.actions_toolcall import (
    format_toolcall_observation_messages,
)


def make_output(
    content: str,
    actions: list[dict],
    cost: float = 1.0,
) -> dict:
    """Create an output dict for DeterministicModel."""
    return {
        "role": "assistant",
        "content": content,
        "extra": {
            "actions": actions,
            "cost": cost,
            "timestamp": time.time(),
        },
    }


def make_toolcall_output(
    content: str | None,
    tool_calls: list[dict],
    actions: list[dict],
) -> dict:
    """Create an output dict for DeterministicToolcallModel."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {
            "actions": actions,
            "cost": 1.0,
            "timestamp": time.time(),
        },
    }


def _process_test_actions(actions: list[dict]) -> bool:
    """Process special test actions and return whether query should retry."""
    for action in actions:
        if "raise" in action:
            raise action["raise"]
        command = action.get("command", "")
        if command.startswith("/sleep "):
            time.sleep(float(command.split("/sleep ")[1]))
            return True
        if command.startswith("/warning"):
            logging.warning(command.split("/warning")[1])
            return True
    return False


class DeterministicModelConfig(BaseModel):
    outputs: list[dict]
    model_name: str = "deterministic"
    observation_template: str = (
        "{% if output.exception_info %}"
        "<exception>{{output.exception_info}}</exception>\n"
        "{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n"
        "<output>\n{{output.output}}</output>"
    )


class DeterministicModel:
    def __init__(self, **kwargs: Any) -> None:
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        return output

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": (
                        f"{self.__class__.__module__}."
                        f"{self.__class__.__name__}"
                    ),
                },
            }
        }


class DeterministicToolcallModelConfig(BaseModel):
    outputs: list[dict]
    model_name: str = "deterministic_toolcall"
    observation_template: str = (
        "{% if output.exception_info %}"
        "<exception>{{output.exception_info}}</exception>\n"
        "{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n"
        "<output>\n{{output.output}}</output>"
    )


class DeterministicToolcallModel:
    def __init__(self, **kwargs: Any) -> None:
        self.config = DeterministicToolcallModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        return output

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": (
                        f"{self.__class__.__module__}."
                        f"{self.__class__.__name__}"
                    ),
                },
            }
        }
