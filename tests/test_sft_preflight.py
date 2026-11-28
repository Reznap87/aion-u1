import json

from aion_u1.training.config import SFTConfig
from aion_u1.training.preflight import check


def fixture(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text('cases:\n  - id: eval.one\n    capability: code.synthesis\n    prompt: benchmark\n    verifier: {type: exact_match, values: ["ok"]}\n')
    caps = tmp_path / "caps.yaml"
    caps.write_text("capabilities:\n  - {id: code.synthesis, name: Code, description: Code, weight: 1.0}\n")
    config = SFTConfig.model_validate({
        "run": {"name": "test", "output_dir": "output"},
        "data": {"train_path": "train.jsonl", "eval_path": "eval.jsonl"},
        "model": {"base_model": "example/model", "revision": "a" * 40, "license_acknowledged": True},
        "peft": {}, "training": {},
    })
    def write(split, prompt, approved=True, group=None):
        row = {"id": split, "group_id": group or split, "capabilities": ["code.synthesis"],
               "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "ok"}],
               "verification": {"status": "verified"},
               "provenance": {"training_approved": approved, "license": "Apache-2.0"}}
        (tmp_path / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
    write("train", "training question")
    write("eval", "validation question")
    return config, suite, caps, write


def test_ready_without_model_download(tmp_path):
    config, suite, caps, _ = fixture(tmp_path)
    result = check(config, tmp_path, suite, caps)
    assert result["ready"] and not result["training_started"]
    assert result["train"]["rows"] == 1


def test_leakage_and_unapproved_data_block(tmp_path):
    config, suite, caps, write = fixture(tmp_path)
    write("eval", "  TRAINING  question ", approved=False, group="train")
    result = check(config, tmp_path, suite, caps)
    assert not result["ready"]
    assert "eval:not_approved" in result["blockers"]
    assert "split_overlap:groups" in result["blockers"]
    assert "split_overlap:prompts" in result["blockers"]


def test_benchmark_overlap_and_unpinned_model_block(tmp_path):
    config, suite, caps, write = fixture(tmp_path)
    config.model.revision = "main"
    write("train", "benchmark")
    result = check(config, tmp_path, suite, caps)
    assert "train:benchmark_overlap" in result["blockers"]
    assert "immutable_model_revision_required" in result["blockers"]


def test_training_entrypoint_blocks_before_model_loading(monkeypatch, tmp_path):
    import builtins
    import pytest
    from aion_u1.training import sft
    from aion_u1.training.data import TrainingDataError

    config, suite, caps, _ = fixture(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "capabilities.yaml").write_bytes(caps.read_bytes())
    config.model.revision = "main"
    monkeypatch.setattr(sft, "find_project_root", lambda: tmp_path)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "transformers", "peft"}:
            pytest.fail("Model loading dependencies must not be imported before preflight")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(TrainingDataError, match="immutable_model_revision_required"):
        sft.run_sft(config, suite=suite)
    assert not (tmp_path / "output").exists()


def test_empty_benchmark_cannot_bypass_preflight(tmp_path):
    import pytest
    config, suite, caps, _ = fixture(tmp_path)
    suite.write_text("cases: []")
    with pytest.raises(ValueError, match="Benchmark suite is empty"):
        check(config, tmp_path, suite, caps)
