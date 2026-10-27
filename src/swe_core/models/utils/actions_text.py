import time

from jinja2 import StrictUndefined, Template


def format_observation_messages(
    outputs: list[dict],
    *,
    observation_template: str,
    template_vars: dict | None = None,
) -> list[dict]:
    """Format execution outputs into user observation messages."""
    results = []
    for output in outputs:
        content = Template(
            observation_template,
            undefined=StrictUndefined,
        ).render(output=output, **(template_vars or {}))
        results.append(
            {
                "role": "user",
                "content": content,
                "extra": {
                    "raw_output": output.get("output", ""),
                    "returncode": output.get("returncode"),
                    "timestamp": time.time(),
                    "exception_info": output.get("exception_info"),
                    **output.get("extra", {}),
                },
            }
        )
    return results
