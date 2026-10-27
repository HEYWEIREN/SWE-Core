"""带人工交互能力的 Agent。"""

import re
import sys
from typing import Any, Literal, NoReturn

from pydantic import Field
from rich.console import Console
from rich.rule import Rule

from swe_core.agents.default import (
    AgentConfig,
    DefaultAgent,
)
from swe_core.agents.utils.prompt_user import (
    multiline_prompt,
    prompt_user,
)
from swe_core.exceptions import (
    LimitsExceeded,
    Submitted,
    TimeExceeded,
    UserInterruption,
)


console = Console(highlight=False)


class InteractiveAgentConfig(AgentConfig):
    """InteractiveAgent 配置。"""

    mode: Literal[
        "human",
        "confirm",
        "yolo",
    ] = "confirm"
    """
    human：直接执行用户输入；
    confirm：执行模型命令前请求确认；
    yolo：不经确认直接执行模型命令。
    """

    whitelist_actions: list[str] = Field(
        default_factory=list
    )
    """匹配这些正则表达式的命令无需确认。"""

    confirm_exit: bool = True
    """Agent 提交任务时是否请求用户确认。"""


class InteractiveAgent(DefaultAgent):
    """在 DefaultAgent 外层加入终端交互。"""

    _MODE_COMMANDS_MAPPING = {
        "/u": "human",
        "/c": "confirm",
        "/y": "yolo",
    }

    def __init__(
        self,
        *args: Any,
        config_class: type[AgentConfig] = (
            InteractiveAgentConfig
        ),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            config_class=config_class,
            **kwargs,
        )

    def _interrupt(
        self,
        content: str,
        *,
        interruption_type: str = (
            "UserInterruption"
        ),
    ) -> NoReturn:
        """用一条用户消息中断当前步骤。"""
        raise UserInterruption(
            {
                "role": "user",
                "content": content,
                "extra": {
                    "interrupt_type":
                        interruption_type,
                },
            }
        )

    def add_messages(
        self,
        *messages: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """在终端显示消息，然后写入消息历史。"""
        for message in messages:
            role = (
                message.get("role")
                or message.get("type")
                or "unknown"
            )
            content = message.get("content") or ""

            if role == "assistant":
                console.print(
                    "\n"
                    "[red][bold]SWE-Core[/bold] "
                    f"(step [bold]{self.n_calls}[/bold], "
                    f"[bold]${self.cost:.2f}[/bold]):"
                    "[/red]\n",
                    end="",
                    highlight=False,
                )
            else:
                console.print(
                    "\n"
                    f"[bold green]"
                    f"{str(role).capitalize()}"
                    f"[/bold green]:\n",
                    end="",
                    highlight=False,
                )

            # 模型内容不能当作 Rich markup 解释。
            console.print(
                content,
                highlight=False,
                markup=False,
            )

        return super().add_messages(*messages)

    def query(self) -> dict[str, Any]:
        """根据当前模式读取用户命令或调用模型。"""
        if self.config.mode == "human":
            command = (
                self._prompt_and_handle_commands(
                    "[bold yellow]>"
                    "[/bold yellow] "
                )
            )

            # 从 human 切换回 confirm/yolo 后，
            # 当前轮直接恢复模型查询。
            if command not in {"/y", "/c"}:
                message = {
                    "role": "user",
                    "content": (
                        "User command:\n"
                        "```bash\n"
                        f"{command}\n"
                        "```"
                    ),
                    "extra": {
                        "actions": [
                            {
                                "command": command,
                            }
                        ],
                    },
                }

                self.add_messages(message)
                return message

        try:
            with console.status(
                "Waiting for the model "
                "to respond..."
            ):
                return super().query()

        except TimeExceeded:
            # 延长 step/cost limit 无法解决墙钟超时，
            # 因此不能再次询问，否则会形成死循环。
            raise

        except LimitsExceeded:
            if not self._stdin_is_interactive():
                # CI、重定向 stdin 或无人值守模式下
                # 不能调用 input()。
                raise

            console.print(
                "Limits exceeded. "
                f"Limits: {self.config.step_limit} "
                "steps, "
                f"${self.config.cost_limit}.\n"
                "Current spend: "
                f"{self.n_calls} steps, "
                f"${self.cost:.2f}."
            )

            self.config.step_limit = int(
                input("New step limit: ")
            )
            self.config.cost_limit = float(
                input("New cost limit: ")
            )

            return super().query()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        """判断当前 stdin 是否为可交互终端。"""
        try:
            return (
                sys.stdin is not None
                and sys.stdin.isatty()
            )
        except (ValueError, OSError):
            return False

    def step(self) -> list[dict[str, Any]]:
        """执行一步，并允许用户用 Ctrl+C 临时介入。"""
        try:
            console.print(Rule())
            return super().step()

        except KeyboardInterrupt:
            user_input = (
                self._prompt_and_handle_commands(
                    "\n\n"
                    "[bold yellow]Interrupted."
                    "[/bold yellow] "
                    "[green]Type a comment or command"
                    "[/green] "
                    "(/h for available commands)\n"
                    "[bold yellow]>"
                    "[/bold yellow] "
                )
            ).strip()

            if (
                not user_input
                or user_input
                in self._MODE_COMMANDS_MAPPING
            ):
                user_input = (
                    "Temporary interruption caught."
                )

            self._interrupt(
                "Interrupted by user: "
                f"{user_input}"
            )

    def execute_actions(
        self,
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """确认并执行 action，保留部分执行结果。"""
        actions = (
            message
            .get("extra", {})
            .get("actions", [])
        )
        commands = [
            action["command"]
            for action in actions
        ]

        outputs: list[dict[str, Any]] = []

        try:
            self._ask_confirmation_or_interrupt(
                commands
            )

            for action in actions:
                outputs.append(
                    self.env.execute(action)
                )

        except Submitted as error:
            self._check_for_new_task_or_submit(
                error
            )

        finally:
            observations = (
                self.model
                .format_observation_messages(
                    message,
                    outputs,
                    self.get_template_vars(),
                )
            )
            result = self.add_messages(
                *observations
            )

        return result

    def _check_for_new_task_or_submit(
        self,
        error: Submitted,
    ) -> NoReturn:
        """确认退出，或将用户的新要求加入消息历史。"""
        if not self.config.confirm_exit:
            raise error

        while True:
            user_input = (
                self._prompt_and_handle_commands(
                    "[bold yellow]"
                    "Agent wants to finish."
                    "[/bold yellow] "
                    "[bold green]Type new task"
                    "[/bold green] or "
                    "[bold]Enter[/bold] to quit "
                    "([bold]/h[/bold] for commands)"
                    "\n"
                    "[bold yellow]>"
                    "[/bold yellow] "
                )
            ).strip()

            if user_input == "/u":
                self._interrupt(
                    "Switched to human mode."
                )

            if (
                user_input
                in self._MODE_COMMANDS_MAPPING
            ):
                # /y 和 /c 只改变模式，然后重新询问
                # 是否真正退出。
                continue

            if user_input:
                self._interrupt(
                    "The user added a new task: "
                    f"{user_input}",
                    interruption_type="UserNewTask",
                )

            raise error

    def _should_ask_confirmation(
        self,
        command: str,
    ) -> bool:
        """判断一条命令是否需要人工确认。"""
        if self.config.mode != "confirm":
            return False

        return not any(
            re.match(pattern, command)
            for pattern
            in self.config.whitelist_actions
        )

    def _ask_confirmation_or_interrupt(
        self,
        commands: list[str],
    ) -> None:
        """请求确认；拒绝时中断当前步骤。"""
        if not any(
            self._should_ask_confirmation(command)
            for command in commands
        ):
            return

        user_input = (
            self._prompt_and_handle_commands(
                "[bold yellow]"
                f"Execute {len(commands)} action(s)?"
                "[/] "
                "[green][bold]Enter[/] to confirm[/], "
                "[red]type [bold]comment[/] "
                "to reject[/], or "
                "[blue][bold]/h[/] for help[/]"
                "\n"
                "[bold yellow]>"
                "[/bold yellow] "
            )
        ).strip()

        if user_input in {"", "/y"}:
            return

        if user_input == "/u":
            self._interrupt(
                "Commands not executed. "
                "Switching to human mode.",
                interruption_type="UserRejection",
            )

        self._interrupt(
            "Commands not executed. "
            "The user rejected the commands "
            f"with this message: {user_input}",
            interruption_type="UserRejection",
        )

    def _prompt_and_handle_commands(
        self,
        prompt: str,
        *,
        multiline: bool = False,
    ) -> str:
        """读取输入，并处理帮助和模式切换命令。"""
        current_prompt = prompt

        while True:
            console.print(
                current_prompt,
                end="",
            )

            if multiline:
                return multiline_prompt()

            user_input = prompt_user()

            if user_input == "/m":
                return multiline_prompt()

            if user_input == "/h":
                console.print(
                    "Current mode: "
                    f"[bold green]"
                    f"{self.config.mode}"
                    f"[/bold green]\n"
                    "[bold green]/y[/bold green] "
                    "switch to yolo mode\n"
                    "[bold green]/c[/bold green] "
                    "switch to confirmation mode\n"
                    "[bold green]/u[/bold green] "
                    "switch to human mode\n"
                    "[bold green]/m[/bold green] "
                    "enter multiline input"
                )
                current_prompt = prompt
                continue

            if (
                user_input
                in self._MODE_COMMANDS_MAPPING
            ):
                new_mode = (
                    self._MODE_COMMANDS_MAPPING[
                        user_input
                    ]
                )

                if self.config.mode == new_mode:
                    current_prompt = (
                        "[bold red]"
                        f"Already in {new_mode} mode."
                        "[/bold red]\n"
                        f"{prompt}"
                    )
                    continue

                self.config.mode = new_mode

                console.print(
                    "Switched to "
                    f"[bold green]{new_mode}"
                    "[/bold green] mode."
                )

            return user_input