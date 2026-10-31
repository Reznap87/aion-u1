from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: str
    seed: int = 42
    resume_from_checkpoint: str | None = None
    overwrite_output_dir: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str
    eval_path: str | None = None
    max_seq_length: int = Field(
        default=4096,
        ge=128,
        le=1_048_576,
    )
    require_training_approval: bool = True


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str
    revision: str = "main"
    trust_remote_code: bool = False
    dtype: Literal[
        "auto",
        "float16",
        "bfloat16",
        "float32",
    ] = "auto"
    attn_implementation: str = "eager"
    license_acknowledged: bool = False


class PeftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    r: int = Field(
        default=16,
        ge=1,
        le=256,
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0,
        le=0.99,
    )
    target_modules: list[str] = Field(
        default_factory=list
    )
    merge_and_save: bool = False


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_train_epochs: float = Field(
        default=1.0,
        gt=0,
    )
    per_device_train_batch_size: int = Field(
        default=1,
        ge=1,
    )
    per_device_eval_batch_size: int = Field(
        default=1,
        ge=1,
    )
    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0,
    )
    warmup_ratio: float = Field(
        default=0.03,
        ge=0,
        le=1,
    )
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
    )
    optim: str = "adamw_torch"
    logging_steps: int = Field(
        default=5,
        ge=1,
    )
    save_strategy: Literal[
        "epoch",
        "steps",
        "no",
    ] = "epoch"
    save_total_limit: int = Field(
        default=2,
        ge=1,
    )
    load_best_model_at_end: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = False
    tf32: bool = False
    full_determinism: bool = False


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    data: DataConfig
    model: ModelConfig
    peft: PeftConfig
    training: TrainingConfig


def find_project_root() -> Path:
    current = Path.cwd().resolve()

    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (
                candidate
                / "src"
                / "aion_u1"
            ).is_dir()
        ):
            return candidate

    return Path(
        __file__
    ).resolve().parents[3]


def load_sft_config(
    path: Path,
) -> SFTConfig:
    raw = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "SFT-Konfiguration muss ein "
            "YAML-Mapping sein."
        )

    return SFTConfig.model_validate(raw)
