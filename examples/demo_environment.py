# demo_environment.py

from swe_core.environments.local import LocalEnvironment
from swe_core.exceptions import Submitted


environment = LocalEnvironment(timeout=2)

commands = [
    "echo hello",
    "pwd",
    "python -c 'import sys; sys.exit(3)'",
    "python -c 'import time; time.sleep(5)'",
]

for command in commands:
    print(f"\n$ {command}")

    result = environment.execute(
        {"command": command}
    )

    print(result)


try:
    environment.execute(
        {
            "command": (
                "printf "
                "'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                "\\ntask finished\\n'"
            )
        }
    )
except Submitted as error:
    print("\nSubmitted:")
    print(error.messages)