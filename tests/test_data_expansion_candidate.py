import importlib.util
import json
from pathlib import Path

import pytest

from aion_u1.training.data import TrainingDataError, is_training_approved
from aion_u1.training.procedural import build as build_procedural
from aion_u1.training.sft import _validate_rows, _validate_split_isolation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_expansion", ROOT / "scripts/generate_data_expansion_candidate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_candidate_is_reproducible_unapproved_and_split(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    module.build(first)
    module.build(second)
    for split in ("train", "dev"):
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
    train, dev = read_rows(first / "train.jsonl"), read_rows(first / "dev.jsonl")
    assert (len(train), len(dev)) == (96, 24)
    assert all(not is_training_approved(row) for row in train + dev)
    with pytest.raises(TrainingDataError):
        _validate_rows(train, require_approval=True)
    _validate_split_isolation(train, dev)
    with pytest.raises(FileExistsError):
        module.build(first)


def test_each_verifier_rejects_wrong_answer():
    for family in module.FAMILIES:
        for split in ("train", "dev"):
            for i in (0, 5):
                _, answer = module.make_case(family, i, split)
                assert module.verify(family, i, split, answer)
                assert not module.verify(family, i, split, "wrong")


def test_candidate_prompt_separation_from_procedural_corpus(tmp_path):
    old = tmp_path / "procedural-v1"
    build_procedural(old, ROOT)
    module.build(tmp_path / "new")
    new = read_rows(tmp_path / "new/train.jsonl") + read_rows(tmp_path / "new/dev.jsonl")
    old_rows = sum((read_rows(old / f"{split}.jsonl") for split in ("train", "dev", "holdout")), [])
    assert {r["messages"][0]["content"] for r in new}.isdisjoint({r["messages"][0]["content"] for r in old_rows})
