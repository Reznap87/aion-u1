import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evaluations/comparison_v2_independent"
spec = importlib.util.spec_from_file_location("comparison_scorer", FOLDER / "scorer.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
generator_spec = importlib.util.spec_from_file_location("comparison_v2", ROOT / "scripts/generate_comparison_v2.py")
generator = importlib.util.module_from_spec(generator_spec)
generator_spec.loader.exec_module(generator)


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_eval_is_pinned_separated_and_breaks_position_heuristics(tmp_path):
    path = FOLDER / "holdout.jsonl"
    manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    cases = read(path)
    assert len(cases) == manifest["cases"] == 18
    relations, forms = Counter(), set()
    prompts, pairs, ids = set(), set(), set()
    first_right = second_right = 0
    for row in cases:
        task, answer = row["messages"]
        prompt = task["content"]
        nums = [int(x) for x in re.findall(r"(?<![\w])-?\d+", prompt)]
        assert len(nums) == 2
        left, right = nums
        assert (left, right) == (row["evaluation"]["left"], row["evaluation"]["right"])
        assert row["evaluation"]["split"] == "holdout"
        assert row["provenance"]["training_approved"] is False
        assert scorer.score(left, right, answer["content"])
        assert not scorer.score(left, right, json.dumps({"groesser": max(left, right) + 1}))
        relations["first" if left > right else "second" if right > left else "equal"] += 1
        forms.add(prompt.split(" ")[0])
        first_right += scorer.score(left, right, json.dumps({"groesser": left}))
        second_right += scorer.score(left, right, json.dumps({"groesser": right}))
        prompts.add(prompt); pairs.add((left, right)); ids.add(row["id"])
    assert len(prompts) == len(pairs) == len(ids) == 18
    assert relations == {"first": 6, "second": 6, "equal": 6}
    assert len(forms) == 3
    assert first_right == second_right == 12  # both miss one third
    assert any(a < 0 and b < 0 for a, b in pairs)
    assert any(a < 0 < b or b < 0 < a for a, b in pairs)
    comparison_data = tmp_path / "comparison-v2"
    generator.build(comparison_data)
    for name in ("train", "dev"):
        prior = comparison_data / f"{name}.jsonl"
        for row in read(prior):
            assert row["messages"][0]["content"] not in prompts
            old_numbers = tuple(map(int, re.findall(r"(?<![\w])-?\d+", row["messages"][0]["content"])))
            assert old_numbers not in pairs and old_numbers[::-1] not in pairs


def test_scorer_rejects_bad_json_and_boolean_values():
    for invalid in ("wrong", "[]", '{"groesser":true}', '{"groesser":4,"extra":1}', '{"groesser":"4"}'):
        assert not scorer.score(4, 2, invalid)
    assert scorer.score(-4, -4, '{"groesser":-4}')
