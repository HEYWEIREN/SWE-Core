from swe_core.utils.serialize import UNSET, recursive_merge


def test_recursive_merge_skips_none_inputs() -> None:
    result = recursive_merge(
        {"alpha": 1},
        None,
        {"beta": 2},
    )

    assert result == {
        "alpha": 1,
        "beta": 2,
    }


def test_recursive_merge_later_values_override_earlier_values() -> None:
    result = recursive_merge(
        {"name": "first", "count": 1},
        {"name": "second"},
    )

    assert result == {
        "name": "second",
        "count": 1,
    }


def test_recursive_merge_merges_nested_dictionaries() -> None:
    result = recursive_merge(
        {
            "config": {
                "timeout": 30,
                "env": {"A": "1"},
            }
        },
        {
            "config": {
                "env": {"B": "2"},
                "cwd": "/tmp/work",
            }
        },
    )

    assert result == {
        "config": {
            "timeout": 30,
            "env": {
                "A": "1",
                "B": "2",
            },
            "cwd": "/tmp/work",
        }
    }


def test_recursive_merge_skips_top_level_unset_values() -> None:
    result = recursive_merge(
        {"keep": "original"},
        {
            "keep": UNSET,
            "missing": UNSET,
        },
    )

    assert result == {"keep": "original"}


def test_recursive_merge_skips_nested_unset_values() -> None:
    result = recursive_merge(
        {
            "config": {
                "keep": "original",
                "unchanged": True,
            }
        },
        {
            "config": {
                "keep": UNSET,
                "missing": UNSET,
                "added": "new",
            }
        },
    )

    assert result == {
        "config": {
            "keep": "original",
            "unchanged": True,
            "added": "new",
        }
    }
