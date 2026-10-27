from typing import Any


UNSET = object()


def recursive_merge(
    *dictionaries: dict | None,
) -> dict[str, Any]:
    """
    递归合并多个字典。

    后面的字典覆盖前面的字典；
    如果两边对应值都是字典，则继续递归合并；
    None  可以是用户明确指定的值
    UNSET 表示“用户根本没有指定，不要覆盖旧值”
    """
    result: dict[str, Any] = {}

    for dictionary in dictionaries:
        if dictionary is None:
            continue

        for key, value in dictionary.items():
            if value is UNSET:
                continue

            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = recursive_merge(
                    result[key],
                    value,
                )
            elif isinstance(value, dict):
                result[key] = recursive_merge(value)
            else:
                result[key] = value

    return result