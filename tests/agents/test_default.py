# tests/agents/test_default.py

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jinja2 import UndefinedError

import swe_core.agents.default as default_module
from swe_core.agents.default import (
    AgentConfig,
    DefaultAgent,
)
from swe_core.exceptions import (
    FormatError,
    Submitted,
)


# ============================================================
# 测试辅助对象
# ============================================================


def make_exit_message(
    status: str,
    submission: str = "",
) -> dict[str, Any]:
    """
    构造 Agent 控制流异常携带的标准 exit 消息。
    """
    return {
        "role": "exit",
        "content": status,
        "extra": {
            "exit_status": status,
            "submission": submission,
        },
    }


def make_assistant_message(
    content: str,
    commands: list[str],
    *,
    cost: float = 0.0,
) -> dict[str, Any]:
    """
    构造带有 actions 的 assistant 消息。
    """
    return {
        "role": "assistant",
        "content": content,
        "extra": {
            "actions": [
                {"command": command}
                for command in commands
            ],
            "cost": cost,
        },
    }


class ScriptedModel:
    """
    DefaultAgent 单元测试使用的可编程假模型。

    events 中的元素可以是：
    - assistant message：query() 正常返回；
    - Exception：query() 直接抛出。

    它比真实 DeterministicModel 更适合测试
    FormatError、未捕获异常等特殊分支。
    """

    def __init__(
        self,
        events: list[dict[str, Any] | BaseException],
        *,
        template_vars: dict[str, Any] | None = None,
    ) -> None:
        self.events = list(events)
        self.template_vars = template_vars or {}

        self.config = {
            "name": "scripted-model",
        }

        self.query_histories: list[
            list[dict[str, Any]]
        ] = []

        self.observation_calls: list[
            tuple[
                dict[str, Any],
                list[dict[str, Any]],
                dict[str, Any],
            ]
        ] = []

    def query(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.query_histories.append(
            deepcopy(messages)
        )

        if not self.events:
            raise RuntimeError(
                "ScriptedModel has no remaining events"
            )

        event = self.events.pop(0)

        if isinstance(event, BaseException):
            raise event

        return deepcopy(event)

    def format_message(
        self,
        *,
        role: str,
        content: str,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "content": content,
            "extra": deepcopy(extra or {}),
        }

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        actual_template_vars = (
            template_vars or {}
        )

        self.observation_calls.append(
            (
                deepcopy(message),
                deepcopy(outputs),
                deepcopy(actual_template_vars),
            )
        )

        return [
            {
                "role": "user",
                "content": json.dumps(
                    output,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "extra": {
                    "message_type": "observation",
                },
            }
            for output in outputs
        ]

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return deepcopy(
            {
                **self.template_vars,
                **kwargs,
            }
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "model": {
                        "name": "scripted-model",
                    }
                }
            }
        }


class RecordingEnvironment:
    """
    不执行真实 shell 的测试环境。

    behaviors 可以指定某条命令：
    - 返回什么 output；
    - 或者抛出什么异常。
    """

    def __init__(
        self,
        *,
        behaviors: dict[
            str,
            dict[str, Any] | BaseException,
        ]
        | None = None,
        template_vars: dict[str, Any] | None = None,
    ) -> None:
        self.behaviors = behaviors or {}
        self.template_vars = template_vars or {}

        self.config = {
            "name": "recording-environment",
        }

        self.actions: list[dict[str, Any]] = []

    def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        self.actions.append(
            deepcopy(action)
        )

        command = action.get("command", "")
        behavior = self.behaviors.get(command)

        if isinstance(behavior, BaseException):
            raise behavior

        if behavior is not None:
            return deepcopy(behavior)

        return {
            "output": f"executed: {command}",
            "returncode": 0,
            "exception_info": "",
        }

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return deepcopy(
            {
                **self.template_vars,
                **kwargs,
            }
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "name":
                            "recording-environment",
                    }
                }
            }
        }


def make_agent(
    model: Any,
    env: Any,
    **overrides: Any,
) -> DefaultAgent:
    """
    创建使用统一默认配置的 DefaultAgent。
    """
    config = {
        "system_template": (
            "System: {{ system_name | default('unknown') }}"
        ),
        "instance_template": (
            "Task={{ task }}; repo={{ repository | default('none') }}"
        ),
        "step_limit": 0,
        "cost_limit": 0.0,
        "wall_time_limit_seconds": 0,
        "max_consecutive_format_errors": 3,
        "output_path": None,
    }

    config.update(overrides)

    return DefaultAgent(
        model=model,
        env=env,
        **config,
    )


# ============================================================
# AgentConfig 测试
# ============================================================


def test_agent_config_has_expected_defaults() -> None:
    config = AgentConfig(
        system_template="system",
        instance_template="task={{ task }}",
    )

    assert config.system_template == "system"
    assert config.instance_template == (
        "task={{ task }}"
    )
    assert config.step_limit == 0
    assert config.cost_limit == 3.0
    assert config.wall_time_limit_seconds == 0
    assert (
        config.max_consecutive_format_errors
        == 3
    )
    assert config.output_path is None


# ============================================================
# 消息管理与模板测试
# ============================================================


def test_add_messages_appends_and_returns_only_new_messages(
) -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    first = {
        "role": "system",
        "content": "system",
    }
    second = {
        "role": "user",
        "content": "task",
    }

    added = agent.add_messages(
        first,
        second,
    )

    assert added == [first, second]
    assert agent.messages == [first, second]

    third = {
        "role": "assistant",
        "content": "answer",
    }

    added_again = agent.add_messages(third)

    assert added_again == [third]
    assert agent.messages == [
        first,
        second,
        third,
    ]


def test_get_template_vars_merges_all_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel(
        [],
        template_vars={
            "model_only": "model",
            "shared": "model",
            "nested": {
                "model": True,
            },
        },
    )

    env = RecordingEnvironment(
        template_vars={
            "env_only": "environment",
            "shared": "environment",
            "nested": {
                "environment": True,
            },
        }
    )

    agent = make_agent(
        model,
        env,
        step_limit=7,
    )

    agent.cost = 1.25
    agent.n_calls = 4
    agent._start_time = 100.0

    agent.extra_template_vars = {
        "extra_only": "extra",
        "shared": "extra",
        "nested": {
            "extra": True,
        },
    }

    monkeypatch.setattr(
        "swe_core.agents.default.time.time",
        lambda: 115.0,
    )

    variables = agent.get_template_vars(
        call_only="call",
        shared="call",
        nested={
            "call": True,
        },
    )

    assert variables["step_limit"] == 7
    assert variables["env_only"] == "environment"
    assert variables["model_only"] == "model"
    assert variables["extra_only"] == "extra"
    assert variables["call_only"] == "call"

    # 后面的来源覆盖前面的来源。
    assert variables["shared"] == "call"

    # 嵌套字典应递归合并，而不是整体覆盖。
    assert variables["nested"] == {
        "environment": True,
        "model": True,
        "extra": True,
        "call": True,
    }

    assert variables["n_model_calls"] == 4
    assert variables["model_cost"] == 1.25
    assert variables["elapsed_seconds"] == 15


def test_render_template_uses_runtime_variables() -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(
            template_vars={
                "cwd": "/workspace/project",
            }
        ),
    )

    agent.extra_template_vars = {
        "task": "fix tests",
    }

    rendered = agent._render_template(
        "{{ task }} in {{ cwd }}"
    )

    assert rendered == (
        "fix tests in /workspace/project"
    )


def test_render_template_rejects_missing_variable(
) -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    with pytest.raises(UndefinedError):
        agent._render_template(
            "{{ variable_that_does_not_exist }}"
        )


# ============================================================
# query、execute_actions、step 测试
# ============================================================


def test_query_records_call_cost_and_message() -> None:
    assistant = make_assistant_message(
        "Run command",
        ["echo hello"],
        cost=1.25,
    )

    model = ScriptedModel([assistant])
    agent = make_agent(
        model,
        RecordingEnvironment(),
    )

    initial_message = {
        "role": "user",
        "content": "task",
    }
    agent.add_messages(initial_message)

    result = agent.query()

    assert result == assistant
    assert agent.n_calls == 1
    assert agent.cost == pytest.approx(1.25)
    assert agent.messages == [
        initial_message,
        assistant,
    ]

    # 模型收到的是调用 query 前的历史。
    assert model.query_histories == [
        [initial_message]
    ]


def test_execute_actions_runs_every_action_in_order(
) -> None:
    model = ScriptedModel([])
    env = RecordingEnvironment(
        behaviors={
            "first": {
                "output": "one",
                "returncode": 0,
                "exception_info": "",
            },
            "second": {
                "output": "two",
                "returncode": 3,
                "exception_info": "",
            },
        }
    )

    agent = make_agent(model, env)

    message = make_assistant_message(
        "Run two commands",
        ["first", "second"],
    )

    observations = agent.execute_actions(
        message
    )

    assert env.actions == [
        {"command": "first"},
        {"command": "second"},
    ]

    assert len(observations) == 2
    assert '"output": "one"' in (
        observations[0]["content"]
    )
    assert '"returncode": 3' in (
        observations[1]["content"]
    )

    # 非零返回码只是 observation，
    # 不应该自动终止 Agent。
    assert observations[1]["role"] == "user"
    assert agent.messages == observations

    assert len(model.observation_calls) == 1

    _, outputs, _ = model.observation_calls[0]

    assert outputs[0]["output"] == "one"
    assert outputs[1]["returncode"] == 3


def test_step_adds_assistant_and_observation_messages(
) -> None:
    assistant = make_assistant_message(
        "Say hello",
        ["echo hello"],
    )

    model = ScriptedModel([assistant])
    env = RecordingEnvironment()

    agent = make_agent(model, env)

    agent.add_messages(
        {
            "role": "system",
            "content": "system",
        },
        {
            "role": "user",
            "content": "task",
        },
    )

    observations = agent.step()

    assert agent.n_calls == 1
    assert env.actions == [
        {"command": "echo hello"}
    ]

    assert len(observations) == 1

    assert [
        message["role"]
        for message in agent.messages
    ] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    assert (
        agent.messages[-2]["content"]
        == "Say hello"
    )
    assert "executed: echo hello" in (
        agent.messages[-1]["content"]
    )


# ============================================================
# run() 正常主循环测试
# ============================================================


def test_run_renders_initial_messages_and_returns_submission(
) -> None:
    submit_error = Submitted(
        make_exit_message(
            "Submitted",
            "task completed",
        )
    )

    model = ScriptedModel(
        [
            make_assistant_message(
                "Inspect",
                ["inspect"],
            ),
            make_assistant_message(
                "Finish",
                ["submit"],
            ),
        ],
        template_vars={
            "system_name": "test-system",
        },
    )

    env = RecordingEnvironment(
        behaviors={
            "submit": submit_error,
        }
    )

    agent = make_agent(model, env)

    result = agent.run(
        "fix the issue",
        repository="demo-repository",
    )

    assert result == {
        "exit_status": "Submitted",
        "submission": "task completed",
    }

    assert agent.n_calls == 2
    assert env.actions == [
        {"command": "inspect"},
        {"command": "submit"},
    ]

    assert (
        agent.messages[0]["role"]
        == "system"
    )
    assert agent.messages[0]["content"] == (
        "System: test-system"
    )

    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"] == (
        "Task=fix the issue; "
        "repo=demo-repository"
    )

    assert agent.messages[-1]["role"] == "exit"
    assert (
        agent.messages[-1]["extra"]
        ["exit_status"]
        == "Submitted"
    )

    assert [
        message["role"]
        for message in agent.messages
    ] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "exit",
    ]


# ============================================================
# 限制条件测试
# ============================================================


def test_run_stops_at_step_limit_before_extra_model_call(
) -> None:
    model = ScriptedModel(
        [
            make_assistant_message(
                "First step",
                ["first"],
            ),
            make_assistant_message(
                "This must not be queried",
                ["second"],
            ),
        ]
    )

    env = RecordingEnvironment()

    agent = make_agent(
        model,
        env,
        step_limit=1,
    )

    result = agent.run("test limit")

    assert result["exit_status"] == (
        "LimitsExceeded"
    )
    assert result["submission"] == ""

    # 只允许真正调用一次模型。
    assert agent.n_calls == 1
    assert len(model.query_histories) == 1
    assert env.actions == [
        {"command": "first"}
    ]

    assert agent.messages[-1]["role"] == "exit"


def test_run_stops_at_cost_limit_before_extra_model_call(
) -> None:
    model = ScriptedModel(
        [
            make_assistant_message(
                "Expensive first step",
                ["first"],
                cost=0.5,
            ),
            make_assistant_message(
                "This must not be queried",
                ["second"],
                cost=0.5,
            ),
        ]
    )

    env = RecordingEnvironment()

    agent = make_agent(
        model,
        env,
        cost_limit=0.5,
    )

    result = agent.run("test cost limit")

    assert result["exit_status"] == (
        "LimitsExceeded"
    )
    assert agent.cost == pytest.approx(0.5)
    assert agent.n_calls == 1
    assert len(model.query_histories) == 1
    assert env.actions == [
        {"command": "first"}
    ]


def test_run_stops_on_wall_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel(
        [
            make_assistant_message(
                "This must not be queried",
                ["command"],
            )
        ]
    )

    agent = make_agent(
        model,
        RecordingEnvironment(),
        wall_time_limit_seconds=5,
    )

    agent._start_time = 100.0

    monkeypatch.setattr(
        "swe_core.agents.default.time.time",
        lambda: 105.0,
    )

    result = agent.run("time limited task")

    assert result == {
        "exit_status": "TimeExceeded",
        "submission": "",
    }

    assert agent.n_calls == 0
    assert model.query_histories == []
    assert agent.messages[-1]["role"] == "exit"


# ============================================================
# FormatError 测试
# ============================================================


def test_format_error_messages_are_added_and_model_retries(
) -> None:
    invalid_assistant = {
        "role": "assistant",
        "content": "Invalid output",
        "extra": {
            "cost": 0.25,
        },
    }

    correction = {
        "role": "user",
        "content": "Please provide a valid action.",
        "extra": {},
    }

    format_error = FormatError(
        invalid_assistant,
        correction,
    )

    submit_error = Submitted(
        make_exit_message(
            "Submitted",
            "recovered",
        )
    )

    model = ScriptedModel(
        [
            format_error,
            make_assistant_message(
                "Corrected output",
                ["submit"],
            ),
        ]
    )

    env = RecordingEnvironment(
        behaviors={
            "submit": submit_error,
        }
    )

    agent = make_agent(model, env)

    result = agent.run("recover from error")

    assert result["exit_status"] == "Submitted"
    assert result["submission"] == "recovered"

    # 发生 FormatError 的 API 调用同样计数。
    assert agent.n_calls == 2

    # query() 未正常返回，因此 run() 从异常消息中补记成本。
    assert agent.cost == pytest.approx(0.25)

    assert invalid_assistant in agent.messages
    assert correction in agent.messages

    # 第二次 query 应看到前一次错误与纠错消息。
    second_history = model.query_histories[1]

    assert invalid_assistant in second_history
    assert correction in second_history


def test_successful_step_resets_consecutive_format_errors(
) -> None:
    first_error = FormatError(
        {
            "role": "assistant",
            "content": "bad output one",
            "extra": {},
        },
        {
            "role": "user",
            "content": "correct format one",
            "extra": {},
        },
    )

    second_error = FormatError(
        {
            "role": "assistant",
            "content": "bad output two",
            "extra": {},
        },
        {
            "role": "user",
            "content": "correct format two",
            "extra": {},
        },
    )

    submit_error = Submitted(
        make_exit_message(
            "Submitted",
            "finished after recovery",
        )
    )

    model = ScriptedModel(
        [
            first_error,
            make_assistant_message(
                "A valid intermediate step",
                ["valid"],
            ),
            second_error,
            make_assistant_message(
                "Submit",
                ["submit"],
            ),
        ]
    )

    env = RecordingEnvironment(
        behaviors={
            "submit": submit_error,
        }
    )

    agent = make_agent(
        model,
        env,
        max_consecutive_format_errors=2,
    )

    result = agent.run("test reset")

    # 如果中间成功步骤没有清零计数，
    # 第二次 FormatError 就会直接触发
    # RepeatedFormatError，无法走到 submit。
    assert result["exit_status"] == "Submitted"
    assert result["submission"] == (
        "finished after recovery"
    )
    assert agent.n_calls == 4

    assert not any(
        message.get("extra", {}).get(
            "exit_status"
        )
        == "RepeatedFormatError"
        for message in agent.messages
    )


def test_repeated_format_errors_stop_agent() -> None:
    first_invalid = {
        "role": "assistant",
        "content": "invalid one",
        "extra": {
            "cost": 0.1,
        },
    }
    first_correction = {
        "role": "user",
        "content": "correction one",
        "extra": {},
    }

    second_invalid = {
        "role": "assistant",
        "content": "invalid two",
        "extra": {
            "cost": 0.2,
        },
    }
    second_correction = {
        "role": "user",
        "content": "correction two",
        "extra": {},
    }

    model = ScriptedModel(
        [
            FormatError(
                first_invalid,
                first_correction,
            ),
            FormatError(
                second_invalid,
                second_correction,
            ),
        ]
    )

    env = RecordingEnvironment()

    agent = make_agent(
        model,
        env,
        max_consecutive_format_errors=2,
    )

    result = agent.run(
        "keep producing invalid output"
    )

    assert result == {
        "exit_status": "RepeatedFormatError",
        "submission": "",
    }

    assert agent.n_calls == 2
    assert agent.cost == pytest.approx(0.3)
    assert env.actions == []

    assert agent.messages[-1]["role"] == "exit"
    assert agent.messages[-1]["content"] == (
        "RepeatedFormatError"
    )

    # 第二次错误消息也必须先保存，
    # 然后才添加 exit。
    assert agent.messages[-3] == second_invalid
    assert agent.messages[-2] == second_correction


# ============================================================
# 未捕获异常测试
# ============================================================


def test_uncaught_exception_is_recorded_saved_and_reraised(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "nested"
        / "trajectory.json"
    )

    model = ScriptedModel(
        [
            ValueError("unexpected failure"),
        ]
    )

    agent = make_agent(
        model,
        RecordingEnvironment(),
        output_path=output_path,
    )

    with pytest.raises(
        ValueError,
        match="unexpected failure",
    ):
        agent.run("crashing task")

    # 异常不能被吞掉，但 exit 信息应进入内存轨迹。
    assert agent.messages[-1]["role"] == "exit"
    assert (
        agent.messages[-1]["extra"]
        ["exit_status"]
        == "ValueError"
    )
    assert (
        agent.messages[-1]["extra"]
        ["exception_str"]
        == "unexpected failure"
    )
    assert "ValueError: unexpected failure" in (
        agent.messages[-1]["extra"]
        ["traceback"]
    )

    # finally 中的 save() 必须已经执行。
    assert output_path.exists()

    saved = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        saved["info"]["exit_status"]
        == "ValueError"
    )
    assert (
        saved["messages"][-1]["extra"]
        ["exception_str"]
        == "unexpected failure"
    )


# ============================================================
# finally 保存行为测试
# ============================================================


def test_run_calls_save_after_every_loop_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_error = Submitted(
        make_exit_message(
            "Submitted",
            "done",
        )
    )

    model = ScriptedModel(
        [
            make_assistant_message(
                "Normal step",
                ["normal"],
            ),
            make_assistant_message(
                "Submit",
                ["submit"],
            ),
        ]
    )

    env = RecordingEnvironment(
        behaviors={
            "submit": submit_error,
        }
    )

    agent = make_agent(model, env)

    saved_paths: list[Path | None] = []

    def fake_save(
        path: Path | None,
        *extra_dicts: dict[str, Any],
    ) -> dict[str, Any]:
        saved_paths.append(path)
        return {}

    monkeypatch.setattr(
        agent,
        "save",
        fake_save,
    )

    result = agent.run("two loop iterations")

    assert result["exit_status"] == "Submitted"

    # 第一轮：normal step
    # 第二轮：submit 中断
    assert saved_paths == [None, None]


# ============================================================
# serialize() 和 save() 测试
# ============================================================


def test_serialize_combines_agent_model_environment_and_extra_data(
) -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    agent.cost = 1.5
    agent.n_calls = 3

    exit_message = make_exit_message(
        "Submitted",
        "final answer",
    )
    agent.messages = [exit_message]

    data = agent.serialize(
        {
            "info": {
                "run_id": "run-001",
            },
            "custom": {
                "enabled": True,
            },
        }
    )

    assert data["info"]["model_stats"] == {
        "instance_cost": 1.5,
        "api_calls": 3,
    }

    assert (
        data["info"]["exit_status"]
        == "Submitted"
    )
    assert (
        data["info"]["submission"]
        == "final answer"
    )

    assert (
        data["info"]["config"]["model"]["name"]
        == "scripted-model"
    )
    assert (
        data["info"]["config"]
        ["environment"]["name"]
        == "recording-environment"
    )

    assert (
        data["info"]["config"]["agent_type"]
        == (
            "swe_core.agents.default."
            "DefaultAgent"
        )
    )

    assert data["info"]["run_id"] == "run-001"
    assert data["custom"]["enabled"] is True
    assert data["messages"] == [exit_message]
    assert data["trajectory_format"] == (
        "SWE-Core-1.1"
    )


def test_serialize_handles_empty_message_history(
) -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    data = agent.serialize()

    assert data["messages"] == []
    assert data["info"]["exit_status"] == ""
    assert data["info"]["submission"] == ""
    assert data["info"]["model_stats"] == {
        "instance_cost": 0.0,
        "api_calls": 0,
    }


def test_save_with_none_returns_data_without_writing_file(
) -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    data = agent.save(None)

    assert isinstance(data, dict)
    assert data["messages"] == []
    assert data["trajectory_format"] == (
        "SWE-Core-1.1"
    )


def test_save_creates_parent_directory_and_writes_utf8_json(
    tmp_path: Path,
) -> None:
    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    agent.messages = [
        {
            "role": "user",
            "content": "修复这个测试",
            "extra": {},
        }
    ]

    output_path = (
        tmp_path
        / "runs"
        / "nested"
        / "trajectory.json"
    )

    returned = agent.save(output_path)

    assert output_path.exists()

    raw_text = output_path.read_text(
        encoding="utf-8"
    )

    # ensure_ascii=False，应直接保留中文。
    assert "修复这个测试" in raw_text

    loaded = json.loads(raw_text)

    assert loaded == returned
    assert (
        loaded["messages"][0]["content"]
        == "修复这个测试"
    )


def test_save_atomically_replaces_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "trajectory.json"

    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )
    agent.messages = [
        {
            "role": "user",
            "content": "new content",
            "extra": {},
        }
    ]

    real_replace = default_module.os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)

        assert source_path.parent == output_path.parent
        replacements.append(
            (source_path, destination_path)
        )
        real_replace(source_path, destination_path)

    monkeypatch.setattr(
        default_module.os,
        "replace",
        record_replace,
    )

    returned = agent.save(output_path)

    assert len(replacements) == 1
    assert replacements[0][1] == output_path
    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == returned


def test_save_failure_preserves_previous_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "trajectory.json"
    previous_data = {
        "previous": True,
    }
    output_path.write_text(
        json.dumps(previous_data),
        encoding="utf-8",
    )

    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )

    def fail_replace(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(
        default_module.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="replace failed",
    ):
        agent.save(output_path)

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == previous_data
    assert list(
        tmp_path.glob(".trajectory.json.*.tmp")
    ) == []


def test_serialization_failure_preserves_previous_trajectory(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "trajectory.json"
    previous_data = {
        "previous": True,
    }
    output_path.write_text(
        json.dumps(previous_data),
        encoding="utf-8",
    )

    agent = make_agent(
        ScriptedModel([]),
        RecordingEnvironment(),
    )
    agent.messages = [
        {
            "role": "user",
            "content": object(),
        }
    ]

    with pytest.raises(
        TypeError,
        match="not JSON serializable",
    ):
        agent.save(output_path)

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == previous_data
    assert list(
        tmp_path.glob(".trajectory.json.*.tmp")
    ) == []
