import importlib.util
import json
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("preflight", Path(__file__).parents[1] / "scripts" / "inference_memory_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def make_shard(root, entries):
    header = json.dumps(entries).encode()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "weights.safetensors" for name in entries}}))


def test_budget_counts_shapes_and_fails_closed_when_either_reserve_is_short(tmp_path):
    make_shard(tmp_path, {"x": {"dtype": "F32", "shape": [4, 3]}})
    budget = preflight.model_budget(tmp_path)
    assert budget["float16_weight_bytes"] == 24
    need = budget["required_free_bytes"]
    assert preflight.assess(budget, {"available_physical_bytes": need, "available_commit_bytes": need})["eligible"]
    assert not preflight.assess(budget, {"available_physical_bytes": need - 1, "available_commit_bytes": need})["eligible"]
    assert not preflight.assess(budget, {"available_physical_bytes": need, "available_commit_bytes": need - 1})["eligible"]


def test_rejects_unexpected_dtype_and_path_escape(tmp_path):
    make_shard(tmp_path, {"x": {"dtype": "I8", "shape": [2]}})
    with pytest.raises(ValueError, match="dtype"):
        preflight.model_budget(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside"}}))
    with pytest.raises(ValueError, match="shard"):
        preflight.model_budget(tmp_path)
