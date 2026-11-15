import hashlib
import json
from pathlib import Path

import pytest

from aion_u1.training.config import find_project_root, load_sft_config
from aion_u1.training.data import TrainingDataError, is_training_approved
from aion_u1.training.procedural import build
from aion_u1.training.sft import _validate_rows, _validate_split_isolation, trainable_parameter_hashes


def test_independent_input_files_cannot_reuse_prompt_or_group():
    def row(prompt, answer, group):
        return {"group_id": group, "messages": [{"role": "user", "content": prompt},
                                                   {"role": "assistant", "content": answer}]}

    train = [row("same prompt", "first answer", "train")]
    with pytest.raises(TrainingDataError, match="prompts"):
        _validate_split_isolation(train, [row("same prompt", "different answer", "eval")])
    with pytest.raises(TrainingDataError, match="gruppen"):
        _validate_split_isolation(train, [row("different prompt", "answer", "train")])
    _validate_split_isolation(train, [row("different prompt", "answer", "eval")])


def test_authored_corpus_has_disjoint_verified_splits(tmp_path):
    folder = tmp_path / "corpus"
    manifest = build(folder, find_project_root())
    assert manifest["prompt_overlap"] == 0
    assert manifest["external_teacher_calls"] == 0
    assert {name: value["rows"] for name, value in manifest["splits"].items()} == {
        "train": 72, "dev": 12, "holdout": 24
    }
    prompts = {}
    for split in ("train", "dev", "holdout"):
        path = folder / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["splits"][split]["sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        prompts[split] = {row["messages"][0]["content"] for row in rows}
        if split != "holdout":
            _validate_rows(rows, require_approval=True)
            assert all(is_training_approved(row) for row in rows)
        else:
            assert all(not is_training_approved(row) for row in rows)
    assert not prompts["train"] & prompts["dev"]
    assert not prompts["train"] & prompts["holdout"]
    assert not prompts["dev"] & prompts["holdout"]
    with pytest.raises(FileExistsError):
        build(folder, find_project_root())


def test_weight_hashes_detect_actual_optimizer_update():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    model = torch.nn.Linear(2, 1)
    initial = trainable_parameter_hashes(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.tensor([[1.0, 2.0]])).square().sum()
    loss.backward()
    optimizer.step()
    assert trainable_parameter_hashes(model) != initial


def test_cpu_pilot_config_keeps_training_and_holdout_separate():
    path = find_project_root() / "training/configs/sft_cpu_procedural.yaml"
    config = load_sft_config(path)
    assert config.training.max_steps == 16
    assert config.data.require_training_approval is True
    assert config.peft.merge_and_save is True
    assert "holdout" not in config.data.train_path
    assert "holdout" not in config.data.eval_path
