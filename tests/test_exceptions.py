import pytest

from swe_core.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
    TimeExceeded,
    UserInterruption,
)


def make_exit_message(status: str) -> dict:
    return {
        "role": "exit",
        "content": status,
        "extra": {
            "exit_status": status,
            "submission": "",
        },
    }


def test_interrupt_agent_flow_stores_messages() -> None:
    message = make_exit_message("Test")

    error = InterruptAgentFlow(message)

    assert error.messages == (message,)


def test_interrupt_can_store_multiple_messages() -> None:
    first = {"role": "assistant", "content": "invalid output"}
    second = {"role": "user", "content": "Please use the bash tool."}

    error = FormatError(first, second)

    assert error.messages == (first, second)


def test_submitted_is_interrupt_agent_flow() -> None:
    error = Submitted(make_exit_message("Submitted"))

    assert isinstance(error, InterruptAgentFlow)


def test_limits_exceeded_is_interrupt_agent_flow() -> None:
    error = LimitsExceeded(make_exit_message("LimitsExceeded"))

    assert isinstance(error, InterruptAgentFlow)


def test_time_exceeded_is_limits_exceeded() -> None:
    error = TimeExceeded(make_exit_message("TimeExceeded"))

    assert isinstance(error, LimitsExceeded)
    assert isinstance(error, InterruptAgentFlow)


def test_user_interruption_is_interrupt_agent_flow() -> None:
    error = UserInterruption(make_exit_message("UserInterruption"))

    assert isinstance(error, InterruptAgentFlow)


def test_base_exception_can_catch_submitted() -> None:
    message = make_exit_message("Submitted")

    with pytest.raises(InterruptAgentFlow) as exception_info:
        raise Submitted(message)

    assert exception_info.value.messages == (message,)