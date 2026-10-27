"""评测运行器共用的 Agent 进度上报。"""

from swe_core.agents.default import DefaultAgent
from swe_core.run.benchmarks.utils.batch_progress import (
    RunBatchProgressManager,
)


class ProgressTrackingAgent(DefaultAgent):
    """继承默认 Agent，在每一步执行前向进度管理器报告状态。"""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})"
        )
        try:
            return super().step()
        finally:
            # 提交通过异常结束步骤，也需要上报最后一次模型调用的费用。
            self.progress_manager.update_instance_cost(
                self.instance_id, self.cost
            )
