"""评测迁移的离线验收：真实 Agent/本地环境，模拟数据集和容器传输。"""

import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from swe_core.config import get_config_from_spec
from swe_core.models.test_models import make_output
from swe_core.run.benchmarks import programbench, swebench, swebench_single
from swe_core.run.benchmarks.utils.batch_progress import (
    RunBatchProgressManager,
)


def local_config(tmp_path):
    return {
        "agent": {
            "system_template": "You are a coding agent.",
            "instance_template": "Task: {{ task }}",
            "cost_limit": 0,
            "step_limit": 3,
        },
        "model": {
            "model_class": "deterministic",
            "outputs": [
                make_output(
                    "Submit",
                    [
                        {
                            "command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\npatch\\n'"
                        }
                    ],
                )
            ],
        },
        "environment": {"environment_class": "local", "cwd": str(tmp_path)},
    }


def install_dataset(monkeypatch, instances):
    module = ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: copy.deepcopy(instances)
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setattr(swebench_single, "load_dataset", module.load_dataset)


def test_parallel_swebench_saves_predictions_trajectories_and_costs(tmp_path):
    instances = [
        {
            "instance_id": f"repo__issue-{index}",
            "problem_statement": "Fix issue",
        }
        for index in range(8)
    ]
    config = local_config(tmp_path)
    original = copy.deepcopy(config)
    report_path = tmp_path / "statuses.yaml"
    progress = RunBatchProgressManager(len(instances), report_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                swebench.process_instance, instance, tmp_path, config, progress
            )
            for instance in instances
        ]
        for future in futures:
            future.result()

    predictions = json.loads((tmp_path / "preds.json").read_text())
    assert set(predictions) == {
        instance["instance_id"] for instance in instances
    }
    for instance_id, prediction in predictions.items():
        assert prediction == {
            "model_name_or_path": "deterministic",
            "instance_id": instance_id,
            "model_patch": "patch\n",
        }
        trajectory = json.loads(
            (tmp_path / instance_id / f"{instance_id}.traj.json").read_text()
        )
        assert trajectory["info"]["exit_status"] == "Submitted"
        assert trajectory["instance_id"] == instance_id
    assert progress.n_completed == 8
    assert sum(progress._instance_costs.values()) == 8.0
    assert (
        len(
            yaml.safe_load(report_path.read_text())[
                "instances_by_exit_status"
            ]["Submitted"]
        )
        == 8
    )
    assert config == original


def test_model_initialization_failure_propagates_after_removing_old_prediction(
    tmp_path,
):
    instance_id = "broken"
    swebench.update_preds_file(
        tmp_path / "preds.json", instance_id, "old", "old patch"
    )
    config = local_config(tmp_path)
    config["model"] = {"model_class": "unknown"}
    progress = RunBatchProgressManager(1)
    # 与原版一致：模型在 try 之前创建，初始化异常交给外层调度器。
    with pytest.raises(ValueError, match="Unknown model type"):
        swebench.process_instance(
            {"instance_id": instance_id, "problem_statement": "Fix"},
            tmp_path,
            config,
            progress,
        )
    assert instance_id not in json.loads((tmp_path / "preds.json").read_text())


def test_swebench_cli_resume_and_redo(monkeypatch, tmp_path):
    install_dataset(
        monkeypatch, [{"instance_id": "issue", "problem_statement": "Fix"}]
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(local_config(tmp_path)))
    output = tmp_path / "output"
    args = ["-c", str(config_path), "-o", str(output)]
    runner = CliRunner()
    result = runner.invoke(swebench.app, args)
    assert result.exit_code == 0, result.exception
    trajectory = output / "issue/issue.traj.json"
    trajectory.write_text("already done")
    assert runner.invoke(swebench.app, args).exit_code == 0
    assert trajectory.read_text() == "already done"
    result = runner.invoke(swebench.app, [*args, "--redo-existing"])
    assert result.exit_code == 0, result.exception
    assert (
        json.loads(trajectory.read_text())["info"]["exit_status"]
        == "Submitted"
    )


def test_single_cli_uses_sorted_index_and_overrides(monkeypatch, tmp_path):
    install_dataset(
        monkeypatch,
        [
            {"instance_id": "z-last", "problem_statement": "Last"},
            {"instance_id": "a-first", "problem_statement": "First"},
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(local_config(tmp_path)))
    output = tmp_path / "single.json"
    result = CliRunner().invoke(
        swebench_single.app,
        [
            "-c",
            str(config_path),
            "-o",
            str(output),
            "--yolo",
            "--exit-immediately",
            "-l",
            "0",
        ],
    )
    assert result.exit_code == 0, result.exception
    trajectory = json.loads(output.read_text())
    assert trajectory["messages"][1]["content"] == "Task: First"
    assert trajectory["info"]["config"]["agent"]["cost_limit"] == 0
    assert trajectory["info"]["exit_status"] == "Submitted"


@pytest.mark.parametrize(
    "selection, error_type", [("missing", KeyError), ("10", IndexError)]
)
def test_single_cli_preserves_upstream_lookup_errors(
    monkeypatch, selection, error_type
):
    install_dataset(monkeypatch, [{"instance_id": "issue"}])
    result = CliRunner().invoke(swebench_single.app, ["-i", selection])
    assert isinstance(result.exception, error_type)


def test_filter_and_shuffle_are_deterministic_and_do_not_mutate_input():
    instances = [{"instance_id": str(index)} for index in range(20)]
    original = copy.deepcopy(instances)
    selected = swebench.filter_instances(
        instances, filter_spec="1", slice_spec="::2", shuffle=True
    )
    assert selected == swebench.filter_instances(
        instances, filter_spec="1", slice_spec="::2", shuffle=True
    )
    assert all(
        instance["instance_id"].startswith("1") for instance in selected
    )
    assert instances == original


@pytest.mark.parametrize("field", ["image_name", "docker_image"])
def test_swebench_explicit_image(field):
    assert (
        swebench.get_swebench_docker_image_name({field: "custom:image"})
        == "custom:image"
    )
    assert swebench.get_swebench_docker_image_name(
        {"instance_id": "ORG__Repo-1"}
    ) == ("docker.io/swebench/sweb.eval.x86_64.org_1776_repo-1:latest")


def test_startup_template_requires_instance_variables(monkeypatch):
    env = SimpleNamespace()
    monkeypatch.setattr(swebench, "get_environment", lambda config: env)
    with pytest.raises(Exception, match="missing"):
        swebench.get_sb_environment(
            {"run": {"env_startup_command": "{{ missing }}"}},
            {"instance_id": "issue"},
        )


def test_progress_concurrent_completion_and_costs(tmp_path):
    progress = RunBatchProgressManager(20, tmp_path / "report.yaml")

    def finish(index):
        instance_id = str(index)
        progress.on_instance_start(instance_id)
        progress.update_instance_cost(instance_id, 1)
        progress.update_instance_cost(instance_id, 2)
        progress.on_instance_end(instance_id, "Submitted")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(finish, range(20)))
    assert progress.n_completed == 20
    assert sum(progress._instance_costs.values()) == 40
    assert (
        len(
            yaml.safe_load((tmp_path / "report.yaml").read_text())[
                "instances_by_exit_status"
            ]["Submitted"]
        )
        == 20
    )


def test_programbench_serialize_removes_raw_output(tmp_path):
    from swe_core.environments.local import LocalEnvironment
    from swe_core.models.test_models import DeterministicModel

    agent = programbench.ProgramBenchAgent(
        DeterministicModel(outputs=[]),
        LocalEnvironment(cwd=str(tmp_path)),
        progress_manager=RunBatchProgressManager(1),
        **local_config(tmp_path)["agent"],
    )
    agent.messages = [
        {
            "role": "tool",
            "extra": {
                "raw_output": "large",
                "observations": [{"raw_output": "large", "output": "small"}],
            },
        }
    ]
    saved = agent.serialize()
    assert saved["messages"][0]["extra"] == {
        "observations": [{"output": "small"}]
    }


@pytest.mark.parametrize("failure", [False, True])
def test_copy_submission_uses_docker_cp(monkeypatch, tmp_path, failure):
    commands = []
    env = SimpleNamespace(
        container_id="container",
        config=SimpleNamespace(executable="docker"),
        execute=lambda action: commands.append(action["command"]),
    )
    destination = tmp_path / "submission.tar.gz"

    def copy_archive(command, **kwargs):
        assert command == [
            "docker",
            "cp",
            "container:/tmp/_submission.tar.gz",
            str(destination),
        ]
        assert kwargs["check"] is True
        if failure:
            raise subprocess.CalledProcessError(1, command)
        destination.write_bytes(b"archive")

    monkeypatch.setattr(programbench.subprocess, "run", copy_archive)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            programbench.copy_submission(env, destination)
    else:
        programbench.copy_submission(env, destination)
        assert destination.read_bytes() == b"archive"
    assert commands == ["tar -czf /tmp/_submission.tar.gz -C /workspace ."]


@pytest.mark.parametrize("name", ["swebench", "programbench"])
def test_default_configs_render_with_project_template_variables(
    tmp_path, name
):
    from swe_core.environments.local import LocalEnvironment
    from swe_core.models.test_models import DeterministicModel
    from swe_core.run.benchmarks.utils.common import ProgressTrackingAgent

    config = get_config_from_spec(f"benchmarks/{name}.yaml")
    agent = ProgressTrackingAgent(
        DeterministicModel(outputs=[]),
        LocalEnvironment(cwd=str(tmp_path)),
        progress_manager=RunBatchProgressManager(1),
        **config["agent"],
    )
    agent.extra_template_vars = {"task": "Fix issue", "instance": {}}
    assert agent._render_template(agent.config.system_template)
    assert agent._render_template(agent.config.instance_template)
    from jinja2 import StrictUndefined, Template

    assert Template(
        config["model"]["observation_template"], undefined=StrictUndefined
    ).render(
        **agent.get_template_vars(),
        output={"output": "ok", "returncode": 0, "exception_info": ""},
    )


@pytest.mark.parametrize(
    "command", ["swebench", "swebench-single", "programbench"]
)
def test_extra_command_help(command):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swe_core.run.utilities.core_extra",
            command,
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_programbench_cli_runs_resumes_and_reports_copy_failure(
    monkeypatch, tmp_path
):
    """验证原版的归档续跑流程，以及复制失败时的轨迹错误记录。"""
    from swe_core.environments.local import LocalEnvironment

    data_module = ModuleType("programbench.utils.load_data")
    data_module.load_all_instances = lambda **kwargs: [
        {"instance_id": "program", "image_name": "example/program"}
    ]
    filter_module = ModuleType("programbench.utils.instance_filters")
    filter_module.filter_instances = swebench.filter_instances
    monkeypatch.setitem(
        sys.modules, "programbench.utils.load_data", data_module
    )
    monkeypatch.setitem(
        sys.modules, "programbench.utils.instance_filters", filter_module
    )

    environments = []

    class FakeDockerEnvironment(LocalEnvironment):
        def execute(self, action, cwd=""):
            if action["command"].startswith("git config"):
                return {"returncode": 0, "output": ""}
            return super().execute(action, cwd)

    def get_environment(config, **kwargs):
        assert config["image"] == "example/program:task_cleanroom_v6"
        env = FakeDockerEnvironment(cwd=str(tmp_path))
        environments.append(env)
        return env

    def copy_submission(env, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")

    monkeypatch.setattr(programbench, "get_environment", get_environment)
    monkeypatch.setattr(programbench, "copy_submission", copy_submission)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(local_config(tmp_path)))
    output = tmp_path / "output"
    args = ["-c", str(config_path), "-o", str(output)]
    runner = CliRunner()
    result = runner.invoke(programbench.app, args)
    assert result.exit_code == 0, result.exception
    archive = output / "program/submission.tar.gz"
    assert archive.read_bytes() == b"archive"
    trajectory = json.loads((output / "program/program.traj.json").read_text())
    assert trajectory["info"]["exit_status"] == "Submitted"
    assert trajectory["instance_id"] == "program"
    assert runner.invoke(programbench.app, args).exit_code == 0
    assert len(environments) == 1

    def fail_copy(env, destination):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(programbench, "copy_submission", fail_copy)
    result = runner.invoke(programbench.app, [*args, "--redo-existing"])
    assert result.exit_code == 0, result.exception
    assert archive.read_bytes() == b"archive"
    trajectory = json.loads((output / "program/program.traj.json").read_text())
    assert trajectory["info"]["submission_copy_error"] == "copy failed"
    assert len(environments) == 2
