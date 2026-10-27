# src/swe_core/models/litellm_model.py

import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from swe_core.exceptions import FormatError
from swe_core.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from swe_core.models.utils.retry import retry


logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    """
    LiteLLM 模型适配器配置。
    """

    model_name: str
    """
    LiteLLM 使用的模型名称。

    推荐带 provider 前缀，例如：
        openai/gpt-4o
        anthropic/claude-sonnet-4-5
    """

    model_kwargs: dict[str, Any] = {}
    """
    额外传递给 litellm.completion() 的参数。

    例如：
        {
            "temperature": 0,
            "max_tokens": 4096,
        }
    """

    cost_tracking: Literal[
        "default",
        "ignore_errors",
    ] = "default"
    """
    cost 统计策略。

    default:
        LiteLLM 无法计算 cost 时直接报错。

    ignore_errors:
        无法计算 cost 时使用 0.0。
    """

    format_error_template: str = "{{ error }}"
    """
    模型 tool call 格式错误时使用的模板。
    创建配置对象时，如果没传这个字段，就使用 "{{ error }}"。
    """

    observation_template: str = (
        "{% if output.exception_info %}"
        "<exception>{{ output.exception_info }}</exception>\n"
        "{% endif %}"
        "<returncode>{{ output.returncode }}</returncode>\n"
        "<output>\n"
        "{{ output.output }}"
        "</output>"
    )
    """
    Environment 执行 action 后，将 output 渲染成
    tool observation message 时使用的模板。
    """


class LitellmModel:
    """
    基于 LiteLLM Chat Completions 的模型适配器。

    核心职责：

        Agent 内部 messages
                ↓
        _prepare_messages_for_api()
                ↓
        litellm.completion()
                ↓
        LiteLLM response
                ↓
        parse_toolcall_actions()
                ↓
        Agent 内部 assistant message
    """

    abort_exceptions: list[type] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(
        self,
        *,
        config_class: Callable[..., LitellmModelConfig] = LitellmModelConfig,
        **kwargs: Any,
    ) -> None:
        """
        创建 LitellmModel。

        kwargs 最终交给 LitellmModelConfig 校验。
        """

        self.config = config_class(**kwargs)

    def _prepare_messages_for_api(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        将 Agent 内部消息转换成可以发送给模型 API 的消息。

        Agent 内部 message 可能包含：

            {
                "role": "assistant",
                "content": "...",
                "tool_calls": [...],
                "extra": {
                    "actions": [...],
                    "cost": ...,
                    "response": ...,
                    "timestamp": ...
                }
            }

        extra 是 swe_core 自己的内部字段，
        OpenAI-compatible API 不认识，所以发送前删除。

        注意：
        这里只创建新的 dict，不修改原始 messages。
        """

        return [
            {
                key: value
                for key, value in message.items()
                if key != "extra"
            }
            for message in messages
        ]

    def _query(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """
        真正调用 LiteLLM。

        LitellmModel 的 provider-specific 逻辑集中在这一层。
        """

        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL],
            **(
                self.config.model_kwargs
                | kwargs
            ),
        )

    def query(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        调用模型，并转换为 swe_core 内部 assistant message。

        完整流程：

            messages
                ↓
            prepare
                ↓
            LiteLLM API
                ↓
            cost
                ↓
            parse actions
                ↓
            assistant message
        """

        # -------------------------------------------------
        # 1. 调用 API，并处理可重试错误
        # -------------------------------------------------

        for attempt in retry(
            logger=logger,
            abort_exceptions=self.abort_exceptions,
        ):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(
                        messages
                    ),
                    **kwargs,
                )

        # -------------------------------------------------
        # 2. API 已成功返回，所以先计算 cost
        # -------------------------------------------------

        cost_output = self._calculate_cost(
            response
        )

        # -------------------------------------------------
        # 3. 解析 tool calls
        # -------------------------------------------------

        try:
            actions = self._parse_actions(
                response
            )

        except FormatError as error:
            # API 请求已经成功，
            # 即使 action 格式错误，这次调用仍可能产生费用。
            #
            # DefaultAgent 捕获 FormatError 后会从这里读取 cost。
            error.messages[0].setdefault(
                "extra",
                {},
            )

            error.messages[0]["extra"].update(
                cost_output
            )

            # trajectory 中还必须保留原始 provider response，
            # 便于后续排查为什么 parser 失败。
            error.messages[0]["extra"][
                "response"
            ] = self._serialize_response(
                response
            )

            raise

        # -------------------------------------------------
        # 4. 转换为 Agent 内部 assistant message
        # -------------------------------------------------

        message = (
            response
            .choices[0]
            .message
            .model_dump()
        )

        message["extra"] = {
            "actions": actions,
            "response": self._serialize_response(
                response
            ),
            **cost_output,
            "timestamp": time.time(),
        }

        return message

    def _calculate_cost(
        self,
        response: Any,
    ) -> dict[str, float]:
        """
        使用 LiteLLM 计算本次模型调用成本。

        返回：

            {
                "cost": 0.001
            }

        而不是直接返回 float，
        这样 query() 可以直接：

            message["extra"] = {
                **cost_output,
                ...
            }
        """

        try:
            cost = (
                litellm
                .cost_calculator
                .completion_cost(
                    response,
                    model=self.config.model_name,
                )
            )

            if cost is None or cost <= 0.0:
                raise ValueError(
                    "Cost must be > 0.0, "
                    f"got {cost}"
                )

        except Exception as error:
            if (
                self.config.cost_tracking
                != "ignore_errors"
            ):
                message = (
                    "Error calculating cost for "
                    f"model {self.config.model_name}: "
                    f"{error}. "
                    "If this model/provider does not "
                    "provide LiteLLM cost metadata, "
                    "set cost_tracking='ignore_errors'."
                )

                logger.critical(message)

                raise RuntimeError(
                    message
                ) from error

            cost = 0.0

        return {
            "cost": float(cost),
        }

    def _parse_actions(
        self,
        response: Any,
    ) -> list[dict[str, Any]]:
        """
        从 LiteLLM response 中解析 bash tool calls。

        具体 JSON/tool name/command 校验逻辑
        已经在 Stage 6 的 parse_toolcall_actions() 中实现，
        这里不重复实现。
        """

        tool_calls = (
            response
            .choices[0]
            .message
            .tool_calls
            or []
        )

        return parse_toolcall_actions(
            tool_calls,
            format_error_template=(
                self.config.format_error_template
            ),
            template_kwargs={
                "finish_reason": (
                    response
                    .choices[0]
                    .finish_reason
                ),
            },
        )

    def _serialize_response(
        self,
        response: Any,
    ) -> Any:
        """
        将 LiteLLM 原始 response 转成适合 trajectory
        JSON 序列化的数据。

        正常情况下 LiteLLM response 是 Pydantic 对象，
        可以 model_dump(mode="json")。

        如果 provider 返回特殊不可序列化对象，
        最后退化为 repr(response)，确保 trajectory
        不因为记录 response 而崩溃。
        """

        try:
            return response.model_dump(
                mode="json"
            )

        except Exception:
            return repr(response)

    def format_message(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        创建普通的 Agent 内部 message。

        当前 Stage 7 不实现 multimodal，
        所以直接返回 kwargs 即可。

        例如：

            format_message(
                role="system",
                content="..."
            )

        返回：

            {
                "role": "system",
                "content": "..."
            }
        """

        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        将 Environment outputs 转成 tool messages。

        这里直接复用 Stage 6，
        LitellmModel 不再实现第二套 observation formatter。
        """

        actions = (
            message
            .get("extra", {})
            .get("actions", [])
        )

        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=(
                self.config.observation_template
            ),
            template_vars=template_vars,
        )

    def get_template_vars(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        将模型配置暴露给 Agent 的 Jinja2 模板系统。
        """

        return self.config.model_dump()

    def serialize(
        self,
    ) -> dict[str, Any]:
        """
        返回可合并进 trajectory 的模型配置。
        """

        return {
            "info": {
                "config": {
                    "model": (
                        self.config.model_dump(
                            mode="json"
                        )
                    ),
                    "model_type": (
                        f"{self.__class__.__module__}."
                        f"{self.__class__.__name__}"
                    ),
                },
            }
        }