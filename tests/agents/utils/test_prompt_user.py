from pathlib import Path
from typing import Any

import swe_core.agents.utils.prompt_user as prompts


class RecordingSession:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[
            tuple[tuple[Any, ...], dict[str, Any]]
        ] = []

    def prompt(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        self.calls.append((args, kwargs))
        return self.result


def test_sessions_share_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history_calls: list[tuple[str, object]] = []
    session_options: list[dict[str, Any]] = []

    def fake_history(path: str) -> object:
        history = object()
        history_calls.append((path, history))
        return history

    def fake_session(**kwargs: Any) -> object:
        session_options.append(kwargs)
        return object()

    monkeypatch.setattr(
        prompts,
        "FileHistory",
        fake_history,
    )
    monkeypatch.setattr(
        prompts,
        "PromptSession",
        fake_session,
    )

    history_path = tmp_path / "nested" / "history.txt"
    prompts._create_prompt_sessions(history_path)

    assert history_path.parent.is_dir()
    assert history_calls[0][0] == str(history_path)
    assert (
        session_options[0]["history"]
        is session_options[1]["history"]
    )
    assert session_options[1]["multiline"] is True


def test_single_and_multiline_prompts(
    monkeypatch,
) -> None:
    single_line = RecordingSession("answer")
    multiline = RecordingSession("line one\nline two")

    monkeypatch.setattr(
        prompts,
        "_get_prompt_sessions",
        lambda: (single_line, multiline),
    )

    assert prompts.prompt_user("> ") == "answer"
    assert prompts.multiline_prompt("> ") == (
        "line one\nline two"
    )
    assert single_line.calls == [(('> ',), {})]

    args, kwargs = multiline.calls[0]
    assert args == ("> ",)
    assert kwargs["bottom_toolbar"] is (
        prompts._MULTILINE_BOTTOM_TOOLBAR
    )
    assert "Esc, then Enter" in (
        prompts._MULTILINE_TOOLBAR_TEXT
    )
    assert "Ctrl+R" in prompts._MULTILINE_TOOLBAR_TEXT
