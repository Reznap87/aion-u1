"""Exercise SFT bookkeeping with fakes, without model imports or training."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from aion_u1.training import sft
from aion_u1.training.config import SFTConfig
from aion_u1.training.data import TrainingDataError


REVISION = "c" * 40
COMMIT = "a" * 40
PRIVATE_ERROR = "private@example.invalid token=synthetic-secret"


class FakeParameter:
    requires_grad = True

    def __init__(self):
        self.value = 1.0

    def numel(self):
        return 1

    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def float(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return struct.pack("<d", self.value)


def _row(prompt="train", group="train"):
    return {
        "id": group,
        "group_id": group,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "synthetic answer"},
        ],
        "verification": {"status": "verified"},
        "provenance": {"training_approved": True, "license": "MIT"},
    }


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    weights = base / "model.safetensors"
    weights.write_bytes(b"synthetic base weights")
    (base / "LICENSE").write_text("synthetic MIT license", encoding="utf-8")
    (base / "source_revision.json").write_text(json.dumps({
        "revision": REVISION,
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(COMMIT, encoding="utf-8")
    _write_rows(tmp_path / "train.jsonl", [_row()])
    _write_rows(tmp_path / "eval.jsonl", [_row("dev", "dev")])
    run = SimpleNamespace(
        root=tmp_path, base=base, stage=None, failure=RuntimeError(PRIVATE_ERROR),
        mutate=True, emit_steps=True, save_weights=True, calls=[], arguments=None,
    )

    def fail_at(stage):
        run.calls.append(stage)
        if run.stage == stage:
            raise run.failure

    class FakeModel:
        dtype = "float32"

        def __init__(self):
            self.config = SimpleNamespace(_commit_hash=None, use_cache=True)
            self.parameter = FakeParameter()

        def parameters(self):
            return [self.parameter]

        def named_parameters(self):
            return [("synthetic.weight", self.parameter)]

        def print_trainable_parameters(self):
            pass

        def enable_input_require_grads(self):
            pass

        def merge_and_unload(self):
            fail_at("merge")
            return self

        def save_pretrained(self, directory):
            fail_at("merged_save")
            save_export(directory, "model.safetensors")

    def save_export(directory, weight_name):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")
        if run.save_weights:
            (directory / weight_name).write_bytes(run.model.parameter.tobytes())

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        chat_template = "synthetic template"

        def apply_chat_template(self, messages, **kwargs):
            return [11] if kwargs["add_generation_prompt"] else [11, 12, 0]

        def save_pretrained(self, directory):
            fail_at("tokenizer_save")
            Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            fail_at("arguments")
            run.arguments = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            fail_at("trainer")
            self.model = kwargs["model"]
            self.callbacks = kwargs["callbacks"]
            self.state = SimpleNamespace(global_step=20)

        def train(self, **kwargs):
            fail_at("train")
            if run.emit_steps:
                for callback in self.callbacks:
                    callback.on_step_begin(None, self.state, None, model=self.model)
            if run.mutate:
                self.model.parameter.value += 1.0
            if run.emit_steps:
                self.state.global_step += 1
                for callback in self.callbacks:
                    callback.on_step_end(None, self.state, None, model=self.model)
            return SimpleNamespace(metrics={"train_loss": 1.0})

        def evaluate(self, metric_key_prefix):
            fail_at(metric_key_prefix)
            return {metric_key_prefix + "_loss": 1.0}

        def save_model(self, directory):
            fail_at("save")
            save_export(directory, "model.safetensors")

    run.model = FakeModel()
    run.trainer_class = FakeTrainer
    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: None
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.utils = SimpleNamespace(data=SimpleNamespace(Dataset=object))
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft.get_peft_model = lambda model, config: model
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: run.model,
    )
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer(),
    )
    transformers.Trainer = FakeTrainer
    transformers.TrainerCallback = object
    transformers.TrainingArguments = FakeTrainingArguments
    for name, module in (("torch", torch), ("peft", peft), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sft, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(sft, "_package_versions", lambda: {"torch": "fake-offline"})
    run.config = SFTConfig.model_validate({
        "run": {"name": "synthetic", "output_dir": "output"},
        "data": {"train_path": "train.jsonl", "eval_path": "eval.jsonl"},
        "model": {"base_model": str(base), "revision": REVISION, "license_acknowledged": True},
        "peft": {"enabled": False},
        "training": {"max_steps": 1, "save_steps": 1, "save_strategy": "steps",
                     "evaluate_before_training": True},
    })
    run.manifest = lambda: json.loads(
        (tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8")
    )
    return run


@pytest.mark.parametrize("stage", [
    "arguments", "trainer", "baseline_dev", "train", "final_dev", "save", "tokenizer_save",
])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_setup_training_and_export_failures_record_private_failed_manifest(fake_run, stage, failure_type):
    fake_run.stage = stage
    fake_run.failure = failure_type(PRIVATE_ERROR)
    with pytest.raises(failure_type) as caught:
        sft.run_sft(fake_run.config)
    assert caught.value is fake_run.failure
    manifest = fake_run.manifest()
    assert manifest["status"] == "failed"
    assert manifest["failure_type"] == failure_type.__name__
    assert failure_type.__name__ in manifest["error"]
    assert PRIVATE_ERROR not in json.dumps(manifest)
    assert "inference_artifacts" not in manifest


@pytest.mark.parametrize("method", ["train", "save_model"])
def test_missing_trainer_methods_cannot_leave_running_manifest(fake_run, monkeypatch, method):
    monkeypatch.delattr(fake_run.trainer_class, method)
    with pytest.raises(AttributeError):
        sft.run_sft(fake_run.config)
    assert fake_run.manifest()["status"] == "failed"
    assert fake_run.manifest()["failure_type"] == "AttributeError"


@pytest.mark.parametrize("field", ["mutate", "emit_steps"])
def test_completion_requires_weight_change_and_steps_in_this_invocation(fake_run, field):
    setattr(fake_run, field, False)
    with pytest.raises(TrainingDataError, match="Gewichtsänderung"):
        sft.run_sft(fake_run.config)
    manifest = fake_run.manifest()
    assert manifest["status"] == "failed"
    assert manifest["training_proof"]["optimizer_steps"] >= 20
    assert "save" not in fake_run.calls


def test_success_keeps_git_weights_and_exact_export_provenance(fake_run):
    output = sft.run_sft(fake_run.config)
    manifest = fake_run.manifest()
    assert manifest["status"] == "completed"
    assert manifest["git_commit"] == COMMIT
    assert manifest["model"]["resolved_revision"] == REVISION
    assert manifest["model"]["weights_sha256"] == sft._local_weights_digest(str(fake_run.base))
    assert manifest["data"]["train_sha256"] == sft._sha256(fake_run.root / "train.jsonl")
    proof = manifest["training_proof"]
    assert proof["optimizer_steps"] == 21
    assert proof["optimizer_steps_this_invocation"] == 1
    assert proof["changed_trainable_tensors"] == 1
    assert proof["initial_hashes"] != proof["final_hashes"]
    assert manifest["baseline_dev_metrics"] == {"baseline_dev_loss": 1.0}
    assert manifest["final_dev_metrics"] == {"final_dev_loss": 1.0}
    assert fake_run.arguments["max_steps"] == fake_run.arguments["save_steps"] == 1
    assert fake_run.arguments["eval_strategy"] == "steps"
    assert fake_run.arguments["report_to"] == []
    exported = output / "final_model"
    assert manifest["inference_artifacts"]["final_model"] == sft.artifact_inventory(exported)
    assert "model.safetensors" in manifest["inference_artifacts"]["final_model"]
    assert (exported / "LICENSE.base-model").read_bytes() == (fake_run.base / "LICENSE").read_bytes()
    notice = json.loads((exported / "AION_TRAINING_NOTICE.json").read_text(encoding="utf-8"))
    assert notice["training_commit"] == COMMIT
    assert notice["base_revision"] == REVISION
    assert notice["training_proof"] == proof
    assert notice["quality_verified"] is False
    assert "LoRA" not in notice["modification"]


def test_merged_export_gets_inventory_license_and_training_notice(fake_run):
    fake_run.config.peft.enabled = True
    fake_run.config.peft.target_modules = ["synthetic"]
    fake_run.config.peft.merge_and_save = True
    output = sft.run_sft(fake_run.config)
    manifest = fake_run.manifest()
    assert manifest["status"] == "completed"
    for name in ("final_model", "merged_model"):
        assert manifest["inference_artifacts"][name] == sft.artifact_inventory(output / name)
        assert "LICENSE.base-model" in manifest["inference_artifacts"][name]
        assert "AION_TRAINING_NOTICE.json" in manifest["inference_artifacts"][name]


def test_interrupted_merge_records_failed_without_export_approval(fake_run):
    fake_run.config.peft.enabled = True
    fake_run.config.peft.target_modules = ["synthetic"]
    fake_run.config.peft.merge_and_save = True
    fake_run.stage = "merged_save"
    fake_run.failure = KeyboardInterrupt(PRIVATE_ERROR)
    with pytest.raises(KeyboardInterrupt):
        sft.run_sft(fake_run.config)
    assert fake_run.manifest()["status"] == "failed"
    assert "inference_artifacts" not in fake_run.manifest()


def test_export_inventory_failure_records_failed_manifest(fake_run, monkeypatch):
    def fail_inventory(directory):
        raise RuntimeError(PRIVATE_ERROR)

    monkeypatch.setattr(sft, "artifact_inventory", fail_inventory)
    with pytest.raises(RuntimeError):
        sft.run_sft(fake_run.config)
    assert fake_run.manifest()["status"] == "failed"
    assert PRIVATE_ERROR not in json.dumps(fake_run.manifest())


def test_export_without_weight_files_cannot_complete(fake_run):
    fake_run.save_weights = False
    with pytest.raises(TrainingDataError, match="Gewichtsdateien"):
        sft.run_sft(fake_run.config)
    assert fake_run.manifest()["status"] == "failed"


def test_failure_to_write_completed_manifest_records_failed_instead(fake_run, monkeypatch):
    write_manifest = sft._write_manifest

    def fail_completion(path, manifest):
        if manifest["status"] == "completed":
            raise OSError(PRIVATE_ERROR)
        write_manifest(path, manifest)

    monkeypatch.setattr(sft, "_write_manifest", fail_completion)
    with pytest.raises(OSError):
        sft.run_sft(fake_run.config)
    assert fake_run.manifest()["status"] == "failed"
    assert PRIVATE_ERROR not in json.dumps(fake_run.manifest())


def test_failure_reporting_does_not_replace_original_exception(fake_run, monkeypatch):
    write_manifest = sft._write_manifest
    fake_run.stage = "train"

    def fail_failure_report(path, manifest):
        if manifest["status"] == "failed":
            raise OSError("synthetic reporting failure")
        write_manifest(path, manifest)

    monkeypatch.setattr(sft, "_write_manifest", fail_failure_report)
    with pytest.raises(RuntimeError) as caught:
        sft.run_sft(fake_run.config)
    assert caught.value is fake_run.failure


def test_overwrite_archives_stale_run_before_new_manifest(fake_run):
    output = fake_run.root / "output"
    output.mkdir()
    old_manifest = {"status": "running", "git_commit": "b" * 40}
    (output / "run_manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")
    fake_run.config.run.overwrite_output_dir = True
    sft.run_sft(fake_run.config)
    stale = json.loads((output / "run_manifest.stale.json").read_text(encoding="utf-8"))
    assert stale["status"] == "failed"
    assert stale["git_commit"] == "b" * 40
    assert "abgebrochen" in stale["error"]
    assert fake_run.manifest()["status"] == "completed"


@pytest.mark.parametrize("violation", ["license", "approval", "verification", "prompt_overlap", "group_overlap"])
def test_existing_rights_and_split_gates_reject_before_trainer(fake_run, violation):
    train = _row()
    dev = _row("dev", "dev")
    if violation == "license":
        fake_run.config.model.license_acknowledged = False
    elif violation == "approval":
        train["provenance"]["training_approved"] = False
    elif violation == "verification":
        train["verification"]["status"] = "unverified"
    elif violation == "prompt_overlap":
        dev["messages"][0]["content"] = train["messages"][0]["content"]
    elif violation == "group_overlap":
        dev["group_id"] = train["group_id"]
    _write_rows(fake_run.root / "train.jsonl", [train])
    _write_rows(fake_run.root / "eval.jsonl", [dev])
    with pytest.raises(TrainingDataError):
        sft.run_sft(fake_run.config)
    assert fake_run.calls == []
    assert not (fake_run.root / "output" / "run_manifest.json").exists()


def test_local_source_revision_rejects_changed_base_weights(fake_run):
    (fake_run.base / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(TrainingDataError, match="Basisgewichte"):
        sft.run_sft(fake_run.config)
    assert fake_run.calls == []
