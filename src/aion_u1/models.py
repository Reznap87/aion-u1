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
        "not_contains",
        "regex",
        "exact_match",
        "json_equals",
    ]
    values: list[str] = Field(default_factory=list)
    expected: Any = None

    @model_validator(mode="after")
    def validate_arguments(self) -> "VerifierSpec":
        if self.type in {"contains_all", "not_contains", "regex"}:
            if not self.values:
                raise ValueError(f"Verifier {self.type} benötigt values")
        if self.type == "json_equals" and self.expected is None:
            raise ValueError("json_equals benötigt expected")
        return self


class AuditCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,95}$")
    capability: str
    prompt: str = Field(min_length=1)
    verifier: VerifierSpec
    tags: list[str] = Field(default_factory=list)


class TeacherResponse(BaseModel):
    model_id: str
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0


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
    score: float = Field(ge=0, le=1)
    passed: bool
    verifier_type: str
    latency_ms: int
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class TeacherScore(BaseModel):
    teacher_alias: str
    overall_score: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    cases: int
    passed: int
    capability_scores: dict[str, float]


class AuditReport(BaseModel):
    schema_version: str = "1.0"
    generated_at: datetime
    suite: str
    suite_path: str
    teacher_resolutions: list[TeacherResolutionRecord]
    results: list[CaseResult]
    scores: dict[str, TeacherScore]
