from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{1,63}$")
    display_name: str
    provider: Literal["openrouter"]
    candidate_model_ids: list[str] = Field(min_length=1)
    model_id_env: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    # "unreviewed" ist die ehrliche Voreinstellung: keine Freigabe ohne
    # dokumentierte Lizenzpruefung (AGENTS.md).
    license: str = "unreviewed"
    base_url_env: str = "OPENROUTER_BASE_URL"
    base_url_default: str = "https://openrouter.ai/api/v1"


class CapabilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,95}$")
    name: str
    description: str
    weight: float = Field(gt=0, le=1)
    parents: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    training_sources: list[str] = Field(default_factory=list)
    runtime_tools: list[str] = Field(default_factory=list)
    evaluation: list[str] = Field(default_factory=list)


class VerifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "contains_all",
        "contains_any",
        "not_contains",
        "regex",
        "exact_match",
        "json_equals",
    ]
    values: list[str] = Field(default_factory=list)
    expected: Any = None
    case_insensitive: bool = True
    partial: bool = False
    pass_threshold: float = Field(default=1.0, gt=0, le=1)
    engine: Literal["legacy", "v2"] = "legacy"

    @model_validator(mode="after")
    def validate_arguments(self) -> "VerifierSpec":
        if self.type in {"contains_all", "contains_any", "not_contains", "regex", "exact_match"}:
            if not self.values:
                raise ValueError(f"Verifier {self.type} benötigt values")
        if self.type == "json_equals" and self.expected is None:
            raise ValueError("json_equals benötigt expected")
        if self.partial and self.type != "contains_all":
            raise ValueError("Partial scoring requires contains_all")
        if self.engine == "legacy" and (self.partial or not self.case_insensitive or self.pass_threshold != 1.0 or self.type == "contains_any"):
            raise ValueError("New scoring options require engine=v2")
        return self


class AuditCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,95}$")
    capability: str
    prompt: str = Field(min_length=1)
    modality: Literal["text", "image"] = "text"
    image_path: str | None = None
    image_mime_type: str = "image/png"
    verifier: VerifierSpec
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_modality(self) -> "AuditCase":
        if self.modality == "image" and not self.image_path:
            raise ValueError(
                f"AuditCase {self.id} benötigt image_path."
            )
        return self


class TeacherResponse(BaseModel):
    model_id: str
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    max_tokens: int | None = None


class TeacherResolutionRecord(BaseModel):
    alias: str
    display_name: str
    provider: str
    model_id: str | None
    status: Literal["verified", "configured", "unverified"]
    source: str
    suggestions: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    teacher_alias: str
    teacher_model_id: str
    case_id: str
    capability: str
    prompt: str
    response: str
    score: float | None = Field(ge=0, le=1)
    passed: bool
    verifier_type: str
    latency_ms: int
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
    max_tokens: int | None = None


class TeacherScore(BaseModel):
    teacher_alias: str
    overall_score: float | None = Field(ge=0, le=1)
    measured_score: float | None = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    cases: int
    passed: int
    errored: int = 0
    truncated: int = 0
    capability_scores: dict[str, float]


class AuditReport(BaseModel):
    schema_version: str = "1.0"
    generated_at: datetime
    suite: str
    suite_path: str
    teacher_resolutions: list[TeacherResolutionRecord]
    results: list[CaseResult]
    scores: dict[str, TeacherScore]
    # Provenienz fuer Definition of Done Punkt 7 (reproduzierbarer
    # Evaluationsbericht): Code-Version, Commit, Config- und Suite-Hashes.
    provenance: dict[str, Any] | None = None
