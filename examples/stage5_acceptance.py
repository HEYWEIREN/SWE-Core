import json
from pathlib import Path
from tempfile import TemporaryDirectory

from swe_core.agents.default import DefaultAgent
from swe_core.environments.local import LocalEnvironment
from swe_core.models.test_models import (
    DeterministicModel,
    make_output,
)


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        trajectory_path = workdir / "trajectory.json"

        model = DeterministicModel(
            outputs=[
                make_output(
                    "Run pwd",
                    [{"command": "pwd"}],
                ),
                make_output(
                    "Run ls",
                    [{"command": "ls"}],
                ),
                make_output(
                    "Submit",
                    [
                        {
                            "command": (
                                "printf "
                                "'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                                "\\nstage 5 complete\\n'"
                            )
                        }
                    ],
                ),
            ]
        )

        agent = DefaultAgent(
            model=model,
            env=LocalEnvironment(cwd=str(workdir)),
            system_template="You are a coding agent.",
            instance_template="Task: {{ task }}",
            step_limit=10,
            cost_limit=0.0,
            output_path=trajectory_path,
        )

        print("Stage 5 acceptance")
        print(f"workdir: {workdir}")
        print(f"trajectory: {trajectory_path}")
        print()

        result = agent.run("run pwd, ls, then submit")
        trajectory = json.loads(
            trajectory_path.read_text(encoding="utf-8")
        )

        print("run() result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()

        print("model calls:")
        print(trajectory["info"]["model_stats"]["api_calls"])
        print()

        print("message flow:")
        for index, message in enumerate(
            trajectory["messages"],
            start=1,
        ):
            role = message["role"]
            content = message.get("content", "")
            print(f"{index}. {role}: {content[:80]!r}")

            actions = message.get("extra", {}).get("actions", [])
            for action in actions:
                print(f"   command: {action['command']}")

            if role == "user" and "<output>" in content:
                print("   observation:")
                print(content)

        print()
        print("trajectory saved:")
        print(trajectory_path.exists())


if __name__ == "__main__":
    main()
