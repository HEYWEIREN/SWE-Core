import swe_core


def test_package_version() -> None:
    assert swe_core.__version__ == "0.1.0"