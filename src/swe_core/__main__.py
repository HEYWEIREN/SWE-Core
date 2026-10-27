"""支持通过 python -m swe_core 启动 CLI。"""

from swe_core.run.core import app


def main() -> None:
    """启动 SWE-Core 命令行应用。"""
    app()


if __name__ == "__main__":
    main()