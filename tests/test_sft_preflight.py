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
