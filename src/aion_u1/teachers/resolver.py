from __future__ import annotations

import difflib
import os
from dataclasses import dataclass

from ..models import TeacherConfig
from .base import TeacherClient


@dataclass(frozen=True)
class TeacherResolution:
    config: TeacherConfig
    model_id: str | None
    status: str
    source: str
    suggestions: tuple[str, ...] = ()


async def resolve_teachers(
    client: TeacherClient,
    configs: list[TeacherConfig],
) -> list[TeacherResolution]:
    catalog = await client.list_models()
    catalog_set = set(catalog)
    resolutions: list[TeacherResolution] = []

    for config in configs:
        env_model = None

        if config.model_id_env:
            env_model = os.getenv(config.model_id_env, "").strip() or None

        if env_model:
            status = (
                "verified"
                if env_model in catalog_set
                else "configured"
            )
            suggestions = () if env_model in catalog_set else (
                difflib.get_close_matches(
                    env_model,
                    catalog,
                    n=5,
                    cutoff=0.25,
                )
            )
            resolutions.append(
                TeacherResolution(
                    config=config,
                    model_id=env_model,
                    status=status,
                    source="environment",
                    suggestions=suggestions,
                )
            )
            continue

        selected = next(
            (
                model_id
                for model_id in config.candidate_model_ids
                if model_id in catalog_set
            ),
            None,
        )

        if selected:
            resolutions.append(
                TeacherResolution(
                    config=config,
                    model_id=selected,
                    status="verified",
                    source="catalog",
                    suggestions=(),
                )
            )
            continue

        suggestions = tuple(
            difflib.get_close_matches(
                config.candidate_model_ids[0],
                catalog,
                n=5,
                cutoff=0.20,
            )
        )

        resolutions.append(
            TeacherResolution(
                config=config,
                model_id=None,
                status="unverified",
                source="none",
                suggestions=suggestions,
            )
        )

    return resolutions
