from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from aion_u1.cli import build_parser
from aion_u1.inference import (
    ArtifactError,
    ChatRequest,
    InferenceInputError,
    LocalModelEngine,
    artifact_inventory,
    load_model_artifact,
)


@pytest.fixture
def exported_run(tmp_path: Path) -> Path:
    """Synthetic files test admission only; they are not trained model weights."""
    model = tmp_path / "merged_model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"synthetic-test-fixture")
    manifest = {
        "status": "completed",
        "git_commit": "a" * 40,
        "config": {
            "model": {"license_acknowledged": True},
            "data": {"require_training_approval": True},
            "peft": {"enabled": True},
        },
        "data": {"train_sha256": "b" * 64},
        "model": {"resolved_revision": "c" * 40},
        "inference_artifacts": {"merged_model": artifact_inventory(model)},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def change_manifest(run: Path, change) -> None:
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_complete_export_is_relocatable_and_does_not_claim_runtime(
    exported_run: Path, capsys
) -> None:
    artifact = load_model_artifact(exported_run)
    assert artifact.directory == exported_run / "merged_model"
    assert len(artifact.sha256) == 64
    args = build_parser().parse_args(["model-check", "--run-dir", str(exported_run)])
    assert args.func(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["artifact_verified"] is True
    assert receipt["runtime_verified"] is False
    assert receipt["quality_verified"] is False
    assert receipt["production_ready"] is False


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(status="running"),
    lambda m: m.update(status="failed"),
    lambda m: m.update(git_commit=None),
    lambda m: m["model"].update(resolved_revision="main"),
    lambda m: m["data"].update(train_sha256="missing"),
    lambda m: m["config"]["model"].update(license_acknowledged=False),
    lambda m: m["config"]["data"].update(require_training_approval=False),
    lambda m: m["config"]["peft"].update(enabled="true"),
    lambda m: m.update(inference_artifacts={}),
    lambda m: m.update(config=[]),
])
def test_incomplete_or_unapproved_run_is_rejected(exported_run: Path, mutation) -> None:
    change_manifest(exported_run, mutation)
    with pytest.raises(ArtifactError):
        load_model_artifact(exported_run)


@pytest.mark.parametrize("filename", [
    "model.safetensors", "config.json", "tokenizer_config.json", "extra.json"
])
def test_modified_and_added_artifacts_are_rejected(exported_run: Path, filename: str) -> None:
    (exported_run / "merged_model" / filename).write_bytes(b"modified")
    with pytest.raises(ArtifactError, match="Run-Manifest"):
        load_model_artifact(exported_run)


def test_missing_weight_is_rejected(exported_run: Path) -> None:
    (exported_run / "merged_model" / "model.safetensors").unlink()
    with pytest.raises(ArtifactError):
        load_model_artifact(exported_run)


def test_adapter_only_export_is_rejected_even_with_matching_hashes(exported_run: Path) -> None:
    directory = exported_run / "merged_model"
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    change_manifest(exported_run, lambda m: m["inference_artifacts"].update(
        merged_model=artifact_inventory(directory)
    ))
    with pytest.raises(ArtifactError, match="LoRA-Adapter"):
        load_model_artifact(exported_run)


def test_full_finetune_uses_final_model(exported_run: Path) -> None:
    (exported_run / "merged_model").rename(exported_run / "final_model")
    def update(manifest):
        manifest["config"]["peft"]["enabled"] = False
        manifest["inference_artifacts"]["final_model"] = (
            manifest["inference_artifacts"].pop("merged_model")
        )
    change_manifest(exported_run, update)
    assert load_model_artifact(exported_run).directory.name == "final_model"


def test_duplicate_manifest_keys_are_rejected(exported_run: Path) -> None:
    path = exported_run / "run_manifest.json"
    text = path.read_text(encoding="utf-8")
    path.write_text('{"status":"failed",' + text[1:], encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_model_artifact(exported_run)


@pytest.mark.parametrize("shard", [
    "../outside.safetensors", "/outside.safetensors",
    "C:\\outside.safetensors", "unlisted.safetensors",
])
def test_shards_cannot_escape_the_verified_export(exported_run: Path, shard: str) -> None:
    directory = exported_run / "merged_model"
    (directory / "model.safetensors").unlink()
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}}), encoding="utf-8"
    )
    change_manifest(exported_run, lambda m: m["inference_artifacts"].update(
        merged_model=artifact_inventory(directory)
    ))
    with pytest.raises(ArtifactError, match="Shardindex"):
        load_model_artifact(exported_run)


def test_all_shards_are_bound_to_manifest(exported_run: Path) -> None:
    directory = exported_run / "merged_model"
    (directory / "model.safetensors").rename(directory / "model-00001-of-00001.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    change_manifest(exported_run, lambda m: m["inference_artifacts"].update(
        merged_model=artifact_inventory(directory)
    ))
    assert load_model_artifact(exported_run).directory == directory

@pytest.fixture
def fake_transformers(monkeypatch):
    """Exercise the runtime's token handling without downloads or torch."""
    calls = []
    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (1, len(values))
        def to(self, device):
            return self
        def __len__(self):
            return len(self.values)
        def __getitem__(self, key):
            value = self.values[key]
            return Tensor(value) if isinstance(key, slice) else value
    class Tokenizer:
        chat_template = "synthetic-template"
        eos_token_id = 99
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return cls()
        def apply_chat_template(self, messages, **kwargs):
            return "synthetic prompt"
        def __call__(self, prompt, **kwargs):
            return {"input_ids": Tensor([1, 2, 3]), "attention_mask": Tensor([1, 1, 1])}
        def decode(self, tokens, **kwargs):
            return "synthetic reply"
    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=256)
        generation_config = SimpleNamespace(eos_token_id=99)
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return cls()
        def to(self, device):
            self.device = device
        def eval(self):
            pass
        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return [Tensor([1, 2, 3, 4, 99])]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model
    ))
    return calls


def test_runtime_is_local_only_and_counts_generated_tokens(
    exported_run: Path, fake_transformers
) -> None:
    engine = LocalModelEngine(load_model_artifact(exported_run), context_window=128)
    request = ChatRequest(
        model="aion-u1", messages=[{"role": "user", "content": "Test"}],
        max_tokens=8, temperature=0,
    )
    result = engine.complete(request)
    assert result.content == "synthetic reply"
    assert (result.prompt_tokens, result.completion_tokens, result.finish_reason) == (3, 2, "stop")
    for _, path, kwargs in fake_transformers[:2]:
        assert path == str(exported_run / "merged_model")
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
    assert fake_transformers[1][2]["use_safetensors"] is True
    generation = fake_transformers[2][1]
    assert generation["max_new_tokens"] == 8
    assert generation["do_sample"] is False
    assert "temperature" not in generation


def test_context_limit_rejects_before_generation(exported_run: Path, fake_transformers) -> None:
    engine = LocalModelEngine(load_model_artifact(exported_run), context_window=128)
    request = ChatRequest(
        model="aion-u1", messages=[{"role": "user", "content": "Test"}], max_tokens=128
    )
    with pytest.raises(InferenceInputError, match="Kontextlimit"):
        engine.complete(request)
    assert len(fake_transformers) == 2


def test_serve_requires_its_own_key_before_loading_any_model(monkeypatch) -> None:
    monkeypatch.delenv("AION_U1_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "teacher-key-is-not-an-inference-key")
    args = build_parser().parse_args(["serve", "--run-dir", "does-not-exist"])
    with pytest.raises(ValueError, match="AION_U1_API_KEY"):
        args.func(args)


def test_serve_passes_explicit_precision_to_engine(monkeypatch, exported_run) -> None:
    import aion_u1.inference as inference
    import aion_u1.server as server_module

    observed = {}

    class Engine:
        model_id = "aion-u1"

        def __init__(self, artifact, **kwargs):
            observed.update(kwargs)

    class Server:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def serve_forever(self):
            pass

    monkeypatch.setenv("AION_U1_API_KEY", "local-test-key")
    monkeypatch.setattr(inference, "LocalModelEngine", Engine)
    monkeypatch.setattr(server_module, "InferenceServer", Server)
    args = build_parser().parse_args(["serve", "--run-dir", str(exported_run), "--dtype", "float16"])
    assert args.func(args) == 0
    assert observed["dtype"] == "float16"
