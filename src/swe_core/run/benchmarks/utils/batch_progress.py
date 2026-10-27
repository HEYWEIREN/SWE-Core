"""批量运行的进度展示、费用汇总和退出状态报告。"""

import collections
import time
from datetime import timedelta
from pathlib import Path
from threading import Lock

import yaml
from rich.console import Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


def _shorten_str(s: str, max_len: int, shorten_left=False) -> str:
    if not shorten_left:
        s = s[: max_len - 3] + "..." if len(s) > max_len else s
    else:
        s = "..." + s[-max_len + 3 :] if len(s) > max_len else s
    return f"{s:<{max_len}}"


class RunBatchProgressManager:
    def __init__(
        self,
        num_instances: int,
        yaml_report_path: Path | None = None,
    ):
        """创建总进度条、实例状态行和可选的 YAML 报告路径。"""

        self._spinner_tasks: dict[str, TaskID] = {}
        # 将数据集实例 ID 映射到 Rich 使用的任务 ID。

        self._lock = Lock()
        # 本项目没有 GLOBAL_MODEL_STATS，按实例汇总 Agent 提供的累计费用。
        self._instance_costs: dict[str, float] = {}
        self._start_time = time.time()
        self._total_instances = num_instances

        self._instances_by_exit_status = collections.defaultdict(list)
        self._main_progress_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn(
                "[progress.description]{task.description} (${task.fields[total_cost]})"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[cyan]{task.fields[eta]}[/cyan]"),
            # 使用五分钟的窗口估算速度。
            speed_estimate_period=60 * 5,
        )
        self._task_progress_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("{task.fields[instance_id]}"),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
        )
        # 每个活动实例在同一个进度条中占据一个任务行。

        self._main_task_id = self._main_progress_bar.add_task(
            "[cyan]Overall Progress",
            total=num_instances,
            total_cost="0.00",
            eta="",
        )

        self.render_group = Group(
            self._main_progress_bar, Table(), self._task_progress_bar
        )
        self._yaml_report_path = yaml_report_path

    @property
    def n_completed(self) -> int:
        return sum(
            len(instances)
            for instances in self._instances_by_exit_status.values()
        )

    def _get_eta_text(self) -> str:
        """根据已完成实例的平均耗时估算剩余时间。"""
        try:
            estimated_remaining = (
                (time.time() - self._start_time)
                / self.n_completed
                * (self._total_instances - self.n_completed)
            )
            return f"eta: {timedelta(seconds=int(estimated_remaining))}"
        except ZeroDivisionError:
            return ""

    def update_exit_status_table(self):
        # 通过重建表格更新退出状态，再替换渲染组中的原表格。
        t = Table()
        t.add_column("Exit Status")
        t.add_column("Count", justify="right", style="bold cyan")
        t.add_column("Most recent instances")
        t.show_header = False
        with self._lock:
            t.show_header = True
            # 按各退出状态的实例数量降序排列。
            sorted_items = sorted(
                self._instances_by_exit_status.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            for status, instances in sorted_items:
                instances_str = _shorten_str(
                    ", ".join(reversed(instances)), 55
                )
                t.add_row(status, str(len(instances)), instances_str)
        assert self.render_group is not None
        self.render_group.renderables[1] = t

    def update_instance_cost(self, instance_id: str, cost: float) -> None:
        """接收 Agent 累计费用，供总进度条汇总。"""
        with self._lock:
            self._instance_costs[instance_id] = cost
        self._update_total_costs()

    def _update_total_costs(self) -> None:
        with self._lock:
            self._main_progress_bar.update(
                self._main_task_id,
                total_cost=f"{sum(self._instance_costs.values()):.2f}",
                eta=self._get_eta_text(),
            )

    def update_instance_status(self, instance_id: str, message: str):
        assert self._task_progress_bar is not None
        assert self._main_progress_bar is not None
        with self._lock:
            self._task_progress_bar.update(
                self._spinner_tasks[instance_id],
                status=_shorten_str(message, 30),
                instance_id=_shorten_str(instance_id, 35, shorten_left=True),
            )
        self._update_total_costs()

    def on_instance_start(self, instance_id: str):
        with self._lock:
            self._spinner_tasks[instance_id] = (
                self._task_progress_bar.add_task(
                    description=f"Task {instance_id}",
                    status="Task initialized",
                    total=None,
                    instance_id=instance_id,
                )
            )

    def on_instance_end(
        self, instance_id: str, exit_status: str | None
    ) -> None:
        with self._lock:
            self._instances_by_exit_status[exit_status].append(instance_id)
            try:
                self._task_progress_bar.remove_task(
                    self._spinner_tasks[instance_id]
                )
            except KeyError:
                pass
            self._main_progress_bar.update(
                TaskID(0), advance=1, eta=self._get_eta_text()
            )
        self.update_exit_status_table()
        self._update_total_costs()
        if self._yaml_report_path is not None:
            self._save_overview_data_yaml(self._yaml_report_path)

    def on_uncaught_exception(
        self, instance_id: str, exception: Exception
    ) -> None:
        self.on_instance_end(
            instance_id, f"Uncaught {type(exception).__name__}"
        )

    def print_report(self) -> None:
        """打印按退出状态分组的实例列表。"""
        for status, instances in self._instances_by_exit_status.items():
            print(f"{status}: {len(instances)}")
            for instance in instances:
                print(f"  {instance}")

    def _get_overview_data(self) -> dict:
        """生成按退出状态分组的报告数据。"""
        return {
            # 转为普通字典后再进行序列化。
            "instances_by_exit_status": dict(self._instances_by_exit_status),
        }

    def _save_overview_data_yaml(self, path: Path) -> None:
        """将实例及退出状态写入 YAML 报告。"""
        with self._lock:
            path.write_text(yaml.dump(self._get_overview_data(), indent=4))
