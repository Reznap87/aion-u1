import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from aion_u1.training.procedural import build as build_procedural
from aion_u1.training.sft import _validate_rows, _validate_split_isolation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comparison_v2", ROOT / "scripts/generate_comparison_v2.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_independent_examination_of_output_and_balanced_relations(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    generator.build(first)
    generator.build(second)
    all_rows = {}
    for split, size in (("train", 36), ("dev", 12)):
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
        all_rows[split] = rows(first / f"{split}.jsonl")
        assert len(all_rows[split]) == size
        counts = Counter()
        negative = Counter()
        for row in all_rows[split]:
            prompt, answer = row["messages"]
            match = re.fullmatch(r'Vergleiche (-?\d+) und (-?\d+)\. Gib nur JSON mit dem Feld "groesser" und dem größeren Wert aus\. Bei Gleichheit gib diesen gemeinsamen Wert aus\.', prompt["content"])
            assert match
            a, b = map(int, match.groups())
            relation = "first" if a > b else "second" if b > a else "equal"
            counts[relation] += 1
            if a < 0 and b < 0: negative[relation] += 1
            assert json.loads(answer["content"]) == {"groesser": sorted((a, b))[-1]}
        assert counts == {"first": size // 3, "second": size // 3, "equal": size // 3}
        assert negative == {"first": size // 6, "second": size // 6, "equal": size // 6}
        _validate_rows(all_rows[split], require_approval=True)
    _validate_split_isolation(all_rows["train"], all_rows["dev"])
    with pytest.raises(FileExistsError): generator.build(first)


def test_disjoint_procedural_holdout_prompts(tmp_path):
    prior = tmp_path / "procedural-v1"
    build_procedural(prior, ROOT)
    generator.build(tmp_path / "new")
    old_prompts = {r["messages"][0]["content"] for r in rows(prior / "holdout.jsonl")}
    for split in ("train", "dev"):
        assert old_prompts.isdisjoint(r["messages"][0]["content"] for r in rows(tmp_path / "new" / f"{split}.jsonl"))
