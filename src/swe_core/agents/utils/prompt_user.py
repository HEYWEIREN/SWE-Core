"""终端单行和多行输入工具。"""

from pathlib import Path

from prompt_toolkit.formatted_text.html import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession

from swe_core.paths import (
    get_interactive_history_path,
)


_MULTILINE_TOOLBAR_TEXT = (
    "Submit message: "
    "<b fg='yellow' bg='black'>"
    "Esc, then Enter"
    "</b> | "
    "Navigate history: "
    "<b fg='yellow' bg='black'>"
    "Arrow Up/Down"
    "</b> | "
    "Search history: "
    "<b fg='yellow' bg='black'>"
    "Ctrl+R"
    "</b>"
)

_MULTILINE_BOTTOM_TOOLBAR = HTML(
    _MULTILINE_TOOLBAR_TEXT
)

PromptSessions = tuple[
    PromptSession,
    PromptSession,
]

_prompt_sessions: PromptSessions | None = None


def _create_prompt_sessions(
    history_path: Path | None = None,
) -> PromptSessions:
    """创建共享同一历史记录的单行和多行会话。"""
    resolved_history_path = (
        history_path
        if history_path is not None
        else get_interactive_history_path()
    )

    resolved_history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = FileHistory(
        str(resolved_history_path)
    )

    single_line_session = PromptSession(
        history=history
    )
    multiline_session = PromptSession(
        history=history,
        multiline=True,
    )

    return (
        single_line_session,
        multiline_session,
    )


def _get_prompt_sessions() -> PromptSessions:
    """延迟创建并缓存终端输入会话。"""
    global _prompt_sessions

    if _prompt_sessions is None:
        _prompt_sessions = (
            _create_prompt_sessions()
        )

    return _prompt_sessions


def prompt_user(prompt: str = "") -> str:
    """读取一条单行用户输入。"""
    single_line_session, _ = (
        _get_prompt_sessions()
    )
    return single_line_session.prompt(prompt)


def multiline_prompt(prompt: str = "") -> str:
    """读取一条带历史记录的多行用户输入。"""
    _, multiline_session = (
        _get_prompt_sessions()
    )

    return multiline_session.prompt(
        prompt,
        bottom_toolbar=(
            _MULTILINE_BOTTOM_TOOLBAR
        ),
    )


__all__ = [
    "multiline_prompt",
    "prompt_user",
]
