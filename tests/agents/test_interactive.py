from copy import deepcopy
from typing import Any

import pytest

import swe_core.agents.interactive as interactive
from swe_core.agents.default import DefaultAgent
from swe_core.agents.interactive import (
    InteractiveAgent,
    InteractiveAgentConfig,
)
from swe_core.exceptions import (
    LimitsExceeded,
    Submitted,
    TimeExceeded,
    UserInterruption,
)
from swe_core.models.test_models import (
    DeterministicModel,
    make_output,
)


class RecordingEnvironment:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.actions: list[dict[str, Any]] = []

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
    ) -> dict[str, Any]:
        self.actions.append(deepcopy(action))

        if action["command"] == "submit":
            raise Submitted(
                {
                    "role": "exit",
                    "content": "done",
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": "done",
                    },
                }
            )

        return {
            "output": action["command"],
            "returncode": 0,
            "exception_info": "",
        }

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {},
                    "environment_type": (
                        "RecordingEnvironment"
                    ),
                }
            }
        }


def make_agent(
    *,
    outputs: list[dict[str, Any]] | None = None,
    environment: RecordingEnvironment | None = None,
    **overrides: Any,
) -> InteractiveAgent:
    config = {
        "system_template": "system",
        "instance_template": "{{ task }}",
        "confirm_exit": True,
        **overrides,
    }

    return InteractiveAgent(
        model=DeterministicModel(
            outputs=outputs or []
        ),
        env=environment or RecordingEnvironment(),
        **config,
    )


def test_config_defaults_are_safe() -> None:
    first = InteractiveAgentConfig(
        system_template="system",
        instance_template="{{ task }}",
    )
    second = InteractiveAgentConfig(
        system_template="system",
        instance_template="{{ task }}",
    )

    first.whitelist_actions.append("^pwd$")

    assert first.mode == "confirm"
    assert first.confirm_exit is True
    assert second.whitelist_actions == []


def test_rejection_is_recorded_and_agent_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = RecordingEnvironment()
    agent = make_agent(
        environment=environment,
        outputs=[
            make_output(
                "first",
                [{"command": "first"}],
            ),
            make_output(
                "finish",
                [{"command": "submit"}],
            ),
        ],
    )
    answers = iter([
        "do not run that",
        "",
        "",
    ])
    monkeypatch.setattr(
        interactive,
        "prompt_user",
        lambda: next(answers),
    )

    result = agent.run("test interaction")

    assert result == {
        "exit_status": "Submitted",
        "submission": "done",
    }
    assert environment.actions == [
        {"command": "submit"}
    ]
    assert any(
        message.get("extra", {}).get(
            "interrupt_type"
        )
        == "UserRejection"
        for message in agent.messages
    )


@pytest.mark.parametrize(
    ("mode", "whitelist"),
    [
        ("yolo", []),
        ("human", []),
        ("confirm", [r"^git status$"]),
    ],
)
def test_confirmation_can_be_skipped(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    whitelist: list[str],
) -> None:
    agent = make_agent(
        mode=mode,
        whitelist_actions=whitelist,
    )

    def fail_if_prompted() -> str:
        raise AssertionError(
            "confirmation was not expected"
        )

    monkeypatch.setattr(
        interactive,
        "prompt_user",
        fail_if_prompted,
    )

    agent._ask_confirmation_or_interrupt(
        ["git status"]
    )


def test_slash_commands_and_multiline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    answers = iter(["/h", "/y"])
    monkeypatch.setattr(
        interactive,
        "prompt_user",
        lambda: next(answers),
    )

    assert (
        agent._prompt_and_handle_commands("> ")
        == "/y"
    )
    assert agent.config.mode == "yolo"

    monkeypatch.setattr(
        interactive,
        "prompt_user",
        lambda: "/m",
    )
    monkeypatch.setattr(
        interactive,
        "multiline_prompt",
        lambda: "line one\nline two",
    )

    assert agent._prompt_and_handle_commands("> ") == (
        "line one\nline two"
    )


def test_human_mode_returns_user_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent(mode="human")
    monkeypatch.setattr(
        interactive,
        "prompt_user",
        lambda: "pwd",
    )

    message = agent.query()

    assert message["role"] == "user"
    assert message["extra"]["actions"] == [
        {"command": "pwd"}
    ]
    assert agent.n_calls == 0


def test_submit_can_exit_or_add_new_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = Submitted(
        {
            "role": "exit",
            "content": "done",
            "extra": {
                "exit_status": "Submitted",
                "submission": "done",
            },
        }
    )

    immediate_agent = make_agent(
        confirm_exit=False
    )
    with pytest.raises(Submitted):
        immediate_agent._check_for_new_task_or_submit(
            submitted
        )

    agent = make_agent()
    monkeypatch.setattr(
        interactive,
        "prompt_user",
        lambda: "also update docs",
    )

    with pytest.raises(UserInterruption) as captured:
        agent._check_for_new_task_or_submit(
            submitted
        )

    message = captured.value.messages[0]
    assert message["extra"]["interrupt_type"] == (
        "UserNewTask"
    )
    assert "also update docs" in message["content"]


def test_keyboard_interrupt_becomes_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()

    def interrupt_step(_self: DefaultAgent) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        DefaultAgent,
        "step",
        interrupt_step,
    )
    monkeypatch.setattr(
        interactive,
        "prompt_user",
        lambda: "please continue carefully",
    )

    with pytest.raises(UserInterruption) as captured:
        agent.step()

    assert "please continue carefully" in (
        captured.value.messages[0]["content"]
    )


def test_non_tty_limits_and_timeouts_do_not_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_limited = make_agent(step_limit=1)
    step_limited.n_calls = 1
    monkeypatch.setattr(
        step_limited,
        "_stdin_is_interactive",
        lambda: False,
    )

    def fail_if_prompted(*args: Any) -> str:
        raise AssertionError("input() must not be called")

    monkeypatch.setattr(
        "builtins.input",
        fail_if_prompted,
    )

    with pytest.raises(LimitsExceeded):
        step_limited.query()

    timed_out = make_agent(
        wall_time_limit_seconds=1
    )
    timed_out._start_time = 0

    with pytest.raises(TimeExceeded):
        timed_out.query()
