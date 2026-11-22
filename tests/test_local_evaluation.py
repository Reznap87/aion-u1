import hashlib
import json
from pathlib import Path

import pytest

from aion_u1.inference import Completion
from aion_u1.training import evaluate_local as evaluation
from aion_u1.training.config import find_project_root, load_sft_config


@pytest.fixture
def local_evaluation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    config = load_sft_config(find_project_root() / "training/configs/sft_cpu_procedural.yaml")
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"synthetic-test-fixture-not-real-weights")
    (base / "source_revision.json").write_text(json.dumps({
        "revision": config.model.revision,
        "model_sha256": hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest(),
    }), encoding="utf-8")
    config.model.base_model = str(base)
    folder = tmp_path / "data/verified/procedural-v1"
    folder.mkdir(parents=True)
    row = {
        "id": "synthetic-holdout", "evaluation": {"split": "holdout", "family": "addition", "a": 1, "b": 2},
        "provenance": {"training_approved": False},
        "messages": [{"role": "user", "content": "1+2"}, {"role": "assistant", "content": "3"}],
    }
    holdout = folder / "holdout.jsonl"
    holdout.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (folder / "data_manifest.json").write_text(json.dumps({
        "splits": {"holdout": {"sha256": hashlib.sha256(holdout.read_bytes()).hexdigest()}}
    }), encoding="utf-8")
    calls = []
    class Engine:
        def __init__(self, artifact, **kwargs):
            self.model_id = kwargs["model_id"]
            calls.append(kwargs)
        def complete(self, request):
            return Completion("3", 4, 1, "stop")
    monkeypatch.setattr(evaluation, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(evaluation, "load_sft_config", lambda path: config)
    monkeypatch.setattr(evaluation, "LocalModelEngine", Engine)
    return tmp_path, config, holdout, calls


def test_same_reserved_holdout_is_scored_without_training_claim(local_evaluation):
    root, config, holdout, calls = local_evaluation
    report = evaluation.evaluate(Path("unused"), "base", root / "report.json")
    assert (report["cases"], report["passed"]) == (1, 1)
    assert report["frontier_quality_verified"] is False
    assert report["external_teacher_calls"] == 0
    assert calls[0]["dtype"] == "float32"


def test_holdout_cannot_be_used_for_checkpoint_selection(local_evaluation):
    root, config, holdout, calls = local_evaluation
    config.data.eval_path = str(holdout)
    with pytest.raises(ValueError, match="training or development"):
        evaluation.evaluate(Path("unused"), "base", root / "report.json")
    assert not calls


def test_modified_holdout_is_rejected_before_loading_model(local_evaluation):
    root, config, holdout, calls = local_evaluation
    holdout.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation.evaluate(Path("unused"), "base", root / "report.json")
    assert not calls
