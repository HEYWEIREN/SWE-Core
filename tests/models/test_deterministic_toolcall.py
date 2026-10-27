from swe_core.environments.local import LocalEnvironment
from swe_core.models.test_models import (
    DeterministicToolcallModel,
    make_toolcall_output,
)


def toolcall_output(command: str = "printf hello") -> dict:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": f'{{"command": "{command}"}}',
            },
        }
    ]
    actions = [{"command": command, "tool_call_id": "call_1"}]
    return make_toolcall_output(None, tool_calls, actions)


def test_model_replays_toolcall_output() -> None:
    model = DeterministicToolcallModel(outputs=[toolcall_output()])

    message = model.query([])

    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["extra"]["actions"] == [
        {"command": "printf hello", "tool_call_id": "call_1"}
    ]


def test_toolcall_environment_round_trip() -> None:
    model = DeterministicToolcallModel(outputs=[toolcall_output()])
    message = model.query([])
    outputs = [
        LocalEnvironment().execute(action)
        for action in message["extra"]["actions"]
    ]

    [observation] = model.format_observation_messages(message, outputs)

    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call_1"
    assert observation["extra"]["returncode"] == 0
    assert "hello" in observation["content"]
