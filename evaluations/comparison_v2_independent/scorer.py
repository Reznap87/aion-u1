"""Deterministic scorer for this evaluation's integer and exact JSON contract."""
import json


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def score(left: int, right: int, response: str) -> bool:
    try:
        value = json.loads(response, object_pairs_hook=_unique_keys)
    except (TypeError, ValueError):
        return False
    if not isinstance(value, dict) or set(value) != {"groesser"}:
        return False
    result = value["groesser"]
    if type(result) is not int:
        return False
    # Compute from signed difference, independently of the dataset's max().
    expected = left if left - right >= 0 else right
    return result == expected
