"""
在终端中逐步浏览 Agent 轨迹。

支持单个 JSON 文件或递归读取目录中的 *.traj.json。
支持快捷键导航和 m 切换元数据；e/E 需要另行安装 jless。
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import typer
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from swe_core.models.utils.content_string import get_content_string
from swe_core.run.utilities.inspector_metadata import (
    format_message_metadata,
    format_trajectory_metadata,
)


def _messages_to_steps(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """按消息角色分步：assistant 或包含 actions 的消息开始新页。"""
    steps = []
    current_step = []
    for message in messages:
        # 初始 system/user 消息在第一页，工具结果跟随发出命令的消息。
        if (
            message.get("extra", {}).get("actions")
            or message.get("role") == "assistant"
        ):
            steps.append(current_step)
            current_step = [message]
        else:
            current_step.append(message)
    if current_step:
        steps.append(current_step)
    return steps


app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class BindingCommandProvider(Provider):
    """将键盘操作提供给 Textual 命令面板。"""

    COMMAND_DESCRIPTIONS = {
        "next_step": "Next step in the current trajectory",
        "previous_step": "Previous step in the current trajectory",
        "first_step": "First step in the current trajectory",
        "last_step": "Last step in the current trajectory",
        "scroll_down": "Scroll down",
        "scroll_up": "Scroll up",
        "next_trajectory": "Next trajectory",
        "previous_trajectory": "Previous trajectory",
        "open_in_jless": "Open the current step in jless",
        "open_in_jless_all": "Open the entire trajectory in jless",
        "toggle_reasoning": "Toggle reasoning content visibility",
        "toggle_metadata": "Toggle trajectory and message metadata",
        "reload": "Reload trajectory file from disk",
        "quit": "Quit the inspector",
    }

    async def discover(self) -> Hits:
        app = self.app
        for binding in app.BINDINGS:
            desc = self.COMMAND_DESCRIPTIONS.get(
                binding.action, binding.description
            )
            yield DiscoveryHit(
                desc, lambda b=binding: app.run_action(b.action)
            )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.app
        for binding in app.BINDINGS:
            desc = self.COMMAND_DESCRIPTIONS.get(
                binding.action, binding.description
            )
            score = matcher.match(desc)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(desc),
                    lambda b=binding: app.run_action(b.action),
                )


class TrajectoryInspector(App):
    """加载轨迹并提供步骤、文件导航和重载功能。"""

    COMMANDS = {BindingCommandProvider}
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "↓"),
        Binding("k,up", "scroll_up", "↑"),
        Binding("L", "next_trajectory", "Traj++"),
        Binding("H", "previous_trajectory", "Traj--"),
        Binding("e", "open_in_jless", "Jless"),
        Binding("E", "open_in_jless_all", "Jless (all)"),
        Binding("r", "toggle_reasoning", "Reasoning"),
        Binding("m", "toggle_metadata", "Metadata"),
        Binding("R", "reload", "Reload"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        trajectory_files: list[Path],
        show_reasoning: bool = True,
    ) -> None:
        css_path = os.environ.get(
            "SWE_CORE_INSPECTOR_STYLE_PATH",
            str(
                Path(__file__).parent.parent.parent
                / "config"
                / "inspector.tcss"
            ),
        )
        self.__class__.CSS = Path(css_path).read_text(encoding="utf-8")

        super().__init__()
        self.show_reasoning = show_reasoning
        self.show_metadata = True
        self.trajectory_files = trajectory_files
        self._i_trajectory = 0
        self._i_step = 0
        self.messages: list[dict[str, Any]] = []
        self.trajectory: dict[str, Any] = {}
        self.steps: list[list[dict[str, Any]]] = []

        if trajectory_files:
            self._load_current_trajectory()

    # 当前步骤和文件索引。

    @property
    def i_step(self) -> int:
        """当前步骤索引。"""
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        """限制步骤索引范围，切换后回到页面顶部。"""
        if value != self._i_step and self.n_steps > 0:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_steps(self) -> int:
        """当前轨迹的步骤数。"""
        return len(self.steps)

    @property
    def i_trajectory(self) -> int:
        """当前轨迹文件索引。"""
        return self._i_trajectory

    @i_trajectory.setter
    def i_trajectory(self, value: int) -> None:
        """限制文件索引范围，并加载目标文件。"""
        if value != self._i_trajectory and self.n_trajectories > 0:
            self._i_trajectory = max(0, min(value, self.n_trajectories - 1))
            self._load_current_trajectory()
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_trajectories(self) -> int:
        """轨迹文件总数。"""
        return len(self.trajectory_files)

    def _load_current_trajectory(self) -> None:
        """加载消息数组或包含 messages 的轨迹对象。"""
        # 文件切换、重载失败时都不能沿用上一份文件的统计。
        self.trajectory = {}
        if not self.trajectory_files:
            self.messages = []
            self.steps = []
            return

        trajectory_file = self.trajectory_files[self.i_trajectory]
        try:
            data = json.loads(trajectory_file.read_text(encoding="utf-8"))

            if isinstance(data, list):
                self.messages = data
            elif isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
                self.trajectory = data
            else:
                raise ValueError("Unrecognized trajectory format")

            if not isinstance(self.messages, list) or not all(
                isinstance(message, dict) for message in self.messages
            ):
                raise ValueError("messages must be a list of objects")

            self.steps = _messages_to_steps(self.messages)
            self._i_step = 0
        except (OSError, ValueError) as e:
            self.trajectory = {}
            self.messages = []
            self.steps = []
            self.notify(
                f"Error loading {trajectory_file.name}: {e}", severity="error"
            )

    @property
    def current_trajectory_name(self) -> str:
        """返回用于标题显示的文件名。"""
        if not self.trajectory_files:
            return "No trajectories"
        return self.trajectory_files[self.i_trajectory].name

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()

    def update_content(self) -> None:
        """渲染当前页，处理 ANSI 颜色和输出中的空字节。"""
        container = self.query_one("#content", Vertical)
        container.remove_children()

        if self.show_metadata and self.trajectory_files:
            container.mount(
                Static(
                    Text(format_trajectory_metadata(self.trajectory)),
                    classes="trajectory-metadata",
                )
            )

        if not self.steps:
            container.mount(
                Static("No trajectory loaded or empty trajectory")
            )
            self.title = "Trajectory Inspector - No Data"
            return

        for message in self.steps[self.i_step]:
            content_str = get_content_string(message)
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message.get("role") or message.get("type") or "unknown"
            message_container.mount(
                Static(role.upper(), classes="message-header")
            )
            if self.show_metadata:
                message_container.mount(
                    Static(
                        Text(format_message_metadata(message)),
                        classes="message-metadata",
                    )
                )
            clean_str = content_str.replace("\x00", "")
            message_container.mount(
                Static(
                    Text.from_ansi(clean_str, no_wrap=False),
                    classes="message-content",
                )
            )
            reasoning = message.get("reasoning_content")
            if (
                reasoning
                and self.show_reasoning
                and role.lower() == "assistant"
            ):
                clean_reasoning = reasoning.replace("\x00", "")
                message_container.mount(
                    Static("REASONING", classes="reasoning-header")
                )
                message_container.mount(
                    Static(
                        Text.from_ansi(clean_reasoning, no_wrap=False),
                        classes="reasoning-content",
                    )
                )

        self.title = (
            f"Trajectory {self.i_trajectory + 1}/{self.n_trajectories} - "
            f"{self.current_trajectory_name} - "
            f"Step {self.i_step + 1}/{self.n_steps}"
        )

    # 导航操作同时支持字母键和方向键。

    def action_next_step(self) -> None:
        self.i_step += 1

    def action_previous_step(self) -> None:
        self.i_step -= 1

    def action_first_step(self) -> None:
        self.i_step = 0

    def action_last_step(self) -> None:
        self.i_step = self.n_steps - 1

    def action_next_trajectory(self) -> None:
        self.i_trajectory += 1

    def action_previous_trajectory(self) -> None:
        self.i_trajectory -= 1

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)

    def _open_in_jless(self, path: Path) -> None:
        """暂停界面，通过外部 jless 查看原始 JSON。"""
        with self.suspend():
            try:
                subprocess.run(["jless", path])
            except FileNotFoundError:
                self.notify(
                    "jless not found. Install jless to use e/E; "
                    "normal browsing does not require it.",
                    severity="error",
                )

    def action_open_in_jless(self) -> None:
        """将当前页写入临时 JSON，查看完成后删除临时文件。"""
        if not self.steps:
            self.notify("No messages to display", severity="warning")
            return
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            try:
                json.dump(
                    self.steps[self.i_step],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:
                f.close()
                temp_path.unlink(missing_ok=True)
                raise

        try:
            self._open_in_jless(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def action_toggle_reasoning(self) -> None:
        self.show_reasoning = not self.show_reasoning
        self.update_content()

    def action_toggle_metadata(self) -> None:
        """切换运行概览和消息元数据，保留当前步骤位置。"""
        self.show_metadata = not self.show_metadata
        self.update_content()

    def action_reload(self) -> None:
        """重新读取文件，尽可能保留当前步骤位置。"""
        saved_step = self._i_step
        self._load_current_trajectory()
        self._i_step = (
            min(saved_step, self.n_steps - 1) if self.n_steps > 0 else 0
        )
        self.update_content()
        if self.steps:
            self.notify("Reloaded")

    def action_open_in_jless_all(self) -> None:
        """使用 jless 查看包含配置和统计的完整轨迹。"""
        if not self.trajectory_files:
            self.notify("No trajectory to display", severity="warning")
            return
        self._open_in_jless(self.trajectory_files[self.i_trajectory])


@app.command(help=__doc__)
def main(
    path: str = typer.Argument(
        ".",
        help="Directory to search for trajectory files or specific trajectory file",
    ),
    reasoning: bool = typer.Option(
        True, "--reasoning/--no-reasoning", help="Show reasoning content"
    ),
) -> None:
    path_obj = Path(path)

    if path_obj.is_file():
        trajectory_files = [path_obj]
    elif path_obj.is_dir():
        trajectory_files = sorted(path_obj.rglob("*.traj.json"))
        if not trajectory_files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
    else:
        raise typer.BadParameter(f"Error: Path '{path}' does not exist")

    inspector = TrajectoryInspector(
        trajectory_files, show_reasoning=reasoning
    )
    inspector.run()


if __name__ == "__main__":
    app()
