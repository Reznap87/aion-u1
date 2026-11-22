"""Load a complete, locally trained model without contacting a Teacher."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ArtifactError(ValueError):
    """The run does not contain a verified standalone model."""


class InferenceInputError(ValueError):
    """A request cannot be represented within the local model's limits."""


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=65_536)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )

    model: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0, le=2)
    stream: bool = False


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]


class InferenceEngine(Protocol):
    model_id: str

    def complete(self, request: ChatRequest) -> Completion: ...


def read_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError("Doppelter JSON-Schlüssel.")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactError("Modell-Metadaten fehlen oder sind ungültig.") from exc
    if not isinstance(value, dict):
        raise ArtifactError("Modell-Metadaten müssen ein JSON-Objekt sein.")
    return value


def artifact_inventory(directory: Path) -> dict[str, str]:
    """Hash the exact export, including weights, tokenizer and configuration."""
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactError("Modellverzeichnis fehlt oder ist ein Symlink.")
    inventory: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ArtifactError("Symlinks sind in Modellartefakten nicht erlaubt.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactError("Modellartefakt enthält eine ungültige Datei.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory[path.relative_to(directory).as_posix()] = digest.hexdigest()
    if not inventory:
        raise ArtifactError("Modellverzeichnis ist leer.")
    return inventory


@dataclass(frozen=True)
class ModelArtifact:
    directory: Path
    sha256: str
    training_commit: str
    base_revision: str


def load_model_artifact(run_dir: Path) -> ModelArtifact:
    """Admission checks do not claim model quality or successful inference."""
    manifest = read_json_object(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed":
        raise ArtifactError("Der Trainingsrun ist nicht abgeschlossen.")
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise ArtifactError("Trainingskonfiguration fehlt.")
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    peft_config = config.get("peft", {})
    if not (
        isinstance(model_config, dict)
        and model_config.get("license_acknowledged") is True
        and isinstance(data_config, dict)
        and data_config.get("require_training_approval") is True
        and isinstance(peft_config, dict)
        and type(peft_config.get("enabled")) is bool
    ):
        raise ArtifactError("Lizenz- oder Trainingsdatenfreigabe fehlt.")
    data = manifest.get("data", {})
    model = manifest.get("model", {})
    commit = manifest.get("git_commit")
    revision = model.get("resolved_revision") if isinstance(model, dict) else None
    train_hash = data.get("train_sha256") if isinstance(data, dict) else None
    for value, size in ((commit, 40), (revision, 40), (train_hash, 64)):
        if not isinstance(value, str) or not re.fullmatch(
            rf"[0-9a-f]{{{size}}}", value
        ):
            raise ArtifactError("Commit, Modellrevision oder Datenhash fehlt.")

    kind = "merged_model" if peft_config["enabled"] else "final_model"
    directory = run_dir / kind
    exports = manifest.get("inference_artifacts", {})
    expected = exports.get(kind) if isinstance(exports, dict) else None
    if not isinstance(expected, dict) or not expected:
        raise ArtifactError(
            "Kein hashgebundenes vollständiges Modell vorhanden. "
            "Für LoRA muss peft.merge_and_save aktiviert sein."
        )
    actual = artifact_inventory(directory)
    if expected != actual:
        raise ArtifactError("Modellartefakte stimmen nicht mit dem Run-Manifest überein.")
    if "adapter_config.json" in actual:
        raise ArtifactError("Ein LoRA-Adapter allein ist kein vollständiges Inferenzmodell.")
    if not (
        "config.json" in actual
        and "tokenizer_config.json" in actual
        and (
            "model.safetensors" in actual
            or "model.safetensors.index.json" in actual
        )
    ):
        raise ArtifactError("Vollständige Konfiguration, Tokenizer oder Safetensors fehlen.")
    if "model.safetensors.index.json" in actual:
        index = read_json_object(directory / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactError("Ungültiger Safetensors-Shardindex.")
        for filename in weight_map.values():
            if (
                not isinstance(filename, str)
                or "/" in filename
                or "\\" in filename
                or filename not in actual
                or not filename.endswith(".safetensors")
            ):
                raise ArtifactError("Shardindex verweist auf nicht verifizierte Gewichte.")
    fingerprint = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ModelArtifact(
        directory=directory,
        sha256=fingerprint,
        training_commit=commit,
        base_revision=revision,
    )


class LocalModelEngine:
    """Transformers runtime restricted to one verified local export."""

    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        model_id: str = "aion-u1",
        context_window: int = 4096,
        device: Literal["cpu", "cuda"] = "cpu",
        dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto",
    ) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]{0,127}", model_id):
            raise ValueError("Ungültige Modellkennung.")
        if not 128 <= context_window <= 1_048_576:
            raise ValueError("Ungültiges Kontextlimit.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Inferenzabhängigkeiten fehlen: pip install -e ".[inference]"'
            ) from exc
        self.model_id = model_id
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(artifact.directory),
            local_files_only=True,
            trust_remote_code=False,
        )
        if not self._tokenizer.chat_template:
            raise ArtifactError("Das exportierte Modell besitzt kein Chat-Template.")
        self._model = AutoModelForCausalLM.from_pretrained(
            str(artifact.directory),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        )
        self._model.to(device)
        self._model.eval()
        self._context_window = context_window
        model_limit = getattr(self._model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            self._context_window = min(context_window, model_limit)

    def complete(self, request: ChatRequest) -> Completion:
        if request.model != self.model_id:
            raise InferenceInputError("Unbekanntes Modell.")
        if request.stream:
            raise InferenceInputError("Streaming wird nicht unterstützt.")
        messages = [message.model_dump() for message in request.messages]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (TypeError, ValueError) as exc:
            raise InferenceInputError("Nachrichten passen nicht zum Chat-Template.") from exc
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + request.max_tokens > self._context_window:
            raise InferenceInputError(
                "Prompt und angeforderte Antwort überschreiten das Kontextlimit."
            )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        generation: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generation["temperature"] = request.temperature
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None:
            generation["pad_token_id"] = eos_id
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        generated = output[0][prompt_tokens:]
        count = len(generated)
        eos_ids = getattr(self._model.generation_config, "eos_token_id", eos_id)
        if not isinstance(eos_ids, list):
            eos_ids = [eos_ids]
        ended = bool(count and int(generated[-1]) in eos_ids)
        return Completion(
            content=self._tokenizer.decode(generated, skip_special_tokens=True),
            prompt_tokens=prompt_tokens,
            completion_tokens=count,
            finish_reason="stop" if ended or count < request.max_tokens else "length",
        )
