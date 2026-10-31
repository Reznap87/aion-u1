from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import (
    SFTConfig,
    find_project_root,
    load_sft_config,
)
from .data import (
    TrainingDataError,
    is_training_approved,
    read_jsonl,
)


def _dependency_error() -> RuntimeError:
    return RuntimeError(
        "Trainingsabhängigkeiten fehlen. "
        "Installiere sie mit:\n"
        '  pip install -e ".[train]"'
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _git_commit(
    root: Path,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}

    for package in (
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "safetensors",
    ):
        try:
            versions[package] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    return versions


def _write_manifest(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    require_approval: bool,
) -> None:
    if not rows:
        raise TrainingDataError(
            "Der Trainingsdatensatz ist leer."
        )

    for index, row in enumerate(rows):
        row_id = str(
            row.get("id") or index
        )

        verification = (
            row.get("verification") or {}
        )

        if verification.get("status") != "verified":
            raise TrainingDataError(
                f"Datensatz {row_id} ist nicht "
                "verifiziert."
            )

        if (
            require_approval
            and not is_training_approved(row)
        ):
            raise TrainingDataError(
                f"Datensatz {row_id} hat keine "
                "explizite Trainingsfreigabe."
            )

        messages = row.get("messages")

        if (
            not isinstance(messages, list)
            or not messages
        ):
            raise TrainingDataError(
                f"Datensatz {row_id} enthält "
                "keine messages."
            )

        if (
            messages[-1].get("role")
            != "assistant"
        ):
            raise TrainingDataError(
                f"Letzte Nachricht von {row_id} "
                "ist kein Assistant-Target."
            )


def run_sft(
    config: SFTConfig,
) -> Path:
    try:
        import torch
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise _dependency_error() from exc

    root = find_project_root()
    output_dir = root / config.run.output_dir
    train_path = root / config.data.train_path

    eval_path = (
        root / config.data.eval_path
        if config.data.eval_path
        else None
    )

    if not train_path.is_file():
        raise TrainingDataError(
            "Trainingsdatei nicht gefunden: "
            f"{train_path}"
        )

    if eval_path and not eval_path.is_file():
        raise TrainingDataError(
            f"Eval-Datei nicht gefunden: {eval_path}"
        )

    if not config.model.license_acknowledged:
        raise TrainingDataError(
            "Setze model.license_acknowledged: true "
            "erst nach Prüfung der Lizenz des "
            "Basis- und Referenzmodells."
        )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not config.run.overwrite_output_dir
        and not config.run.resume_from_checkpoint
    ):
        raise TrainingDataError(
            "Ausgabeverzeichnis ist nicht leer: "
            f"{output_dir}. Nutze einen neuen Run "
            "oder setze overwrite_output_dir."
        )

    train_rows = read_jsonl(train_path)
    eval_rows = (
        read_jsonl(eval_path)
        if eval_path
        else []
    )

    _validate_rows(
        train_rows,
        require_approval=(
            config.data.require_training_approval
        ),
    )

    if eval_rows:
        _validate_rows(
            eval_rows,
            require_approval=(
                config.data.require_training_approval
            ),
        )

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )

    torch.manual_seed(config.run.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            config.run.seed
        )

    cuda_available = torch.cuda.is_available()

    if (
        config.training.bf16
        and not cuda_available
    ):
        raise TrainingDataError(
            "bf16 benötigt eine CUDA-GPU."
        )

    if (
        config.training.fp16
        and not cuda_available
    ):
        raise TrainingDataError(
            "fp16 benötigt eine CUDA-GPU."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        revision=config.model.revision,
        trust_remote_code=(
            config.model.trust_remote_code
        ),
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise TrainingDataError(
                "Tokenizer besitzt weder PAD- "
                "noch EOS-Token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    if not tokenizer.chat_template:
        raise TrainingDataError(
            "Das Basismodell muss ein Chat-Template "
            "besitzen. Wähle ein instruct-/chat-fähiges "
            "Basismodell."
        )

    model_kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": (
            config.model.trust_remote_code
        ),
        "low_cpu_mem_usage": True,
        "attn_implementation": (
            config.model.attn_implementation
        ),
    }

    if config.model.dtype == "auto":
        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = getattr(
            torch,
            config.model.dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        **model_kwargs,
    )

    resolved_revision = getattr(
        model.config,
        "_commit_hash",
        None,
    )

    if config.peft.enabled:
        if not config.peft.target_modules:
            raise TrainingDataError(
                "peft.target_modules darf bei "
                "aktiviertem LoRA nicht leer sein."
            )

        peft_config = LoraConfig(
            r=config.peft.r,
            lora_alpha=config.peft.lora_alpha,
            lora_dropout=config.peft.lora_dropout,
            target_modules=(
                config.peft.target_modules
            ),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(
            model,
            peft_config,
        )
        model.print_trainable_parameters()

    if config.training.gradient_checkpointing:
        model.config.use_cache = False

        if hasattr(
            model,
            "enable_input_require_grads",
        ):
            model.enable_input_require_grads()

    eos_token_id = tokenizer.eos_token_id

    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    def token_ids(value: Any) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()

        if (
            value
            and isinstance(
                value[0],
                list,
            )
        ):
            value = value[0]

        return [
            int(token)
            for token in value
        ]

    class TokenizedDataset(
        torch.utils.data.Dataset
    ):
        def __init__(
            self,
            rows: list[dict[str, Any]],
        ) -> None:
            self.items: list[
                dict[str, list[int]]
            ] = []
            self.alignment_mismatches = 0

            for row in rows:
                messages = row["messages"]
                prefix_messages = messages[:-1]

                try:
                    prefix_ids = token_ids(
                        tokenizer.apply_chat_template(
                            prefix_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                        )
                    )

                    full_ids = token_ids(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=False,
                        )
                    )
                except Exception as exc:
                    raise TrainingDataError(
                        "Chat-Template-Fehler in "
                        f"{row.get('id')}: {exc}"
                    ) from exc

                if (
                    eos_token_id is not None
                    and (
                        not full_ids
                        or full_ids[-1] != eos_token_id
                    )
                ):
                    full_ids.append(eos_token_id)

                common_prefix = 0
                limit = min(
                    len(prefix_ids),
                    len(full_ids),
                )

                while (
                    common_prefix < limit
                    and prefix_ids[common_prefix]
                    == full_ids[common_prefix]
                ):
                    common_prefix += 1

                if (
                    common_prefix
                    != len(prefix_ids)
                ):
                    self.alignment_mismatches += 1

                prefix_length = common_prefix

                if (
                    len(full_ids)
                    > config.data.max_seq_length
                ):
                    removed = (
                        len(full_ids)
                        - config.data.max_seq_length
                    )

                    full_ids = full_ids[
                        -config.data.max_seq_length:
                    ]

                    prefix_length = max(
                        0,
                        prefix_length - removed,
                    )

                labels = list(full_ids)

                for index in range(
                    min(
                        prefix_length,
                        len(labels),
                    )
                ):
                    labels[index] = -100

                if all(
                    label == -100
                    for label in labels
                ):
                    raise TrainingDataError(
                        "Kein trainierbares "
                        "Assistant-Target in "
                        f"{row.get('id')}."
                    )

                self.items.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [
                            1
                        ] * len(full_ids),
                        "labels": labels,
                    }
                )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(
            self,
            index: int,
        ) -> dict[str, list[int]]:
            return self.items[index]

    train_dataset = TokenizedDataset(train_rows)
    eval_dataset = (
        TokenizedDataset(eval_rows)
        if eval_rows
        else None
    )

    def collate_features(
        features: list[
            dict[str, list[int]]
        ],
    ) -> dict[str, Any]:
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        pad_token_id = tokenizer.pad_token_id
        batch: dict[str, Any] = {}

        for key in (
            "input_ids",
            "attention_mask",
            "labels",
        ):
            values: list[list[int]] = []

            for feature in features:
                item = list(feature[key])
                padding = max_length - len(item)

                if key == "labels":
                    item.extend(
                        [-100] * padding
                    )
                elif key == "attention_mask":
                    item.extend([0] * padding)
                else:
                    item.extend(
                        [pad_token_id] * padding
                    )

                values.append(item)

            batch[key] = torch.tensor(
                values,
                dtype=torch.long,
            )

        return batch

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    effective_config = config.model_dump(
        mode="json"
    )

    (
        output_dir / "effective_config.yaml"
    ).write_text(
        yaml.safe_dump(
            effective_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    manifest_path = output_dir / "run_manifest.json"

    manifest: dict[str, Any] = {
        "status": "running",
        "config": effective_config,
        "git_commit": _git_commit(root),
        "package_versions": _package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "accelerator": {
            "cuda_available": cuda_available,
            "cuda_device_count": (
                torch.cuda.device_count()
                if cuda_available
                else 0
            ),
        },
        "data": {
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "train_rows": len(train_rows),
            "eval_path": (
                str(eval_path)
                if eval_path
                else None
            ),
            "eval_sha256": (
                _sha256(eval_path)
                if eval_path
                else None
            ),
            "eval_rows": len(eval_rows),
        },
        "model": {
            "base_model": config.model.base_model,
            "requested_revision": (
                config.model.revision
            ),
            "resolved_revision": resolved_revision,
            "dtype": str(model.dtype),
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    }

    _write_manifest(
        manifest_path,
        manifest,
    )

    arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": (
            config.run.overwrite_output_dir
        ),
        "num_train_epochs": (
            config.training.num_train_epochs
        ),
        "per_device_train_batch_size": (
            config.training.per_device_train_batch_size
        ),
        "per_device_eval_batch_size": (
            config.training.per_device_eval_batch_size
        ),
        "gradient_accumulation_steps": (
            config.training.gradient_accumulation_steps
        ),
        "learning_rate": (
            config.training.learning_rate
        ),
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": (
            config.training.lr_scheduler_type
        ),
        "max_grad_norm": (
            config.training.max_grad_norm
        ),
        "optim": config.training.optim,
        "logging_steps": config.training.logging_steps,
        "logging_first_step": True,
        "save_strategy": (
            config.training.save_strategy
        ),
        "save_total_limit": (
            config.training.save_total_limit
        ),
        "save_safetensors": True,
        "eval_strategy": (
            "epoch"
            if eval_dataset
            else "no"
        ),
        "load_best_model_at_end": bool(
            eval_dataset
            and config.training.load_best_model_at_end
        ),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": (
            config.training.gradient_checkpointing
        ),
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "tf32": (
            config.training.tf32
            and cuda_available
        ),
        "full_determinism": (
            config.training.full_determinism
        ),
        "seed": config.run.seed,
        "data_seed": config.run.seed,
        "report_to": [],
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": cuda_available,
        "ddp_find_unused_parameters": False,
        "logging_dir": str(
            output_dir / "logs"
        ),
    }

    if config.training.gradient_checkpointing:
        arguments[
            "gradient_checkpointing_kwargs"
        ] = {
            "use_reentrant": False,
        }

    if not eval_dataset:
        arguments[
            "load_best_model_at_end"
        ] = False

    if (
        arguments["load_best_model_at_end"]
        and config.training.save_strategy == "no"
    ):
        arguments[
            "load_best_model_at_end"
        ] = False

    training_arguments = TrainingArguments(
        **arguments
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_features,
        processing_class=tokenizer,
    )

    result = trainer.train(
        resume_from_checkpoint=(
            config.run.resume_from_checkpoint
        )
    )

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)

    merged_dir: str | None = None

    if (
        config.peft.enabled
        and config.peft.merge_and_save
    ):
        merged_model = model.merge_and_unload()
        merged_model.config.use_cache = True

        merged_path = output_dir / "merged_model"

        merged_model.save_pretrained(
            str(merged_path)
        )
        tokenizer.save_pretrained(
            str(merged_path)
        )

        merged_dir = str(merged_path)

    manifest["status"] = "completed"
    manifest["metrics"] = result.metrics
    manifest["token_alignment_mismatches"] = (
        train_dataset.alignment_mismatches
        + (
            eval_dataset.alignment_mismatches
            if eval_dataset
            else 0
        )
    )
    manifest["artifacts"] = {
        "final_model": str(final_model_dir),
        "merged_model": merged_dir,
    }

    _write_manifest(
        manifest_path,
        manifest,
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-u1-train-sft",
        description=(
            "Trainiert AION-U1 mit einem "
            "verifizierten SFT-Datensatz."
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "training/configs/sft_lora.yaml"
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        config_path = Path(
            args.config
        ).expanduser().resolve()

        config = load_sft_config(config_path)
        run_sft(config)

        return 0

    except (
        TrainingDataError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(
            f"Fehler: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
