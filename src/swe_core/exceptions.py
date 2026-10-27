from typing import Any


Message = dict[str, Any]


class InterruptAgentFlow(Exception):
    """正常中断 Agent 控制循环，并向消息历史追加消息。"""

    def __init__(self, *messages: Message):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Agent 已经完成任务并提交最终结果。"""


class LimitsExceeded(InterruptAgentFlow):
    """Agent 超过步数限制或成本限制。"""


class TimeExceeded(LimitsExceeded):
    """Agent 超过总运行时间限制。"""


class UserInterruption(InterruptAgentFlow):
    """用户主动中断 Agent。"""


class FormatError(InterruptAgentFlow):
    """模型输出不符合 Agent 所要求的格式。""" 