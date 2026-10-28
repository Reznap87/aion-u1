from __future__ import annotations

from pathlib import Path

import yaml

from .models import CapabilitySpec


class CapabilityRegistry:
    def __init__(self, specs: list[CapabilitySpec]):
        self._specs: dict[str, CapabilitySpec] = {}

        for spec in specs:
            if spec.id in self._specs:
                raise ValueError(f"Doppelte Capability: {spec.id}")
            self._specs[spec.id] = spec

        self._validate_dependencies()

    @classmethod
    def load(cls, path: Path) -> "CapabilityRegistry":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise ValueError("Capability-Konfiguration muss ein Mapping sein")

        entries = raw.get("capabilities")

        if not isinstance(entries, list):
            raise ValueError("capabilities muss eine Liste sein")

        specs = [CapabilitySpec.model_validate(entry) for entry in entries]
        return cls(specs)

    def get(self, capability_id: str) -> CapabilitySpec:
        try:
            return self._specs[capability_id]
        except KeyError as exc:
            raise ValueError(
                f"Unbekannte Capability: {capability_id}"
            ) from exc

    def list(self) -> list[CapabilitySpec]:
        return list(self._specs.values())

    @property
    def total_weight(self) -> float:
        return sum(spec.weight for spec in self._specs.values())

    def _validate_dependencies(self) -> None:
        for spec in self._specs.values():
            for parent in spec.parents:
                if parent not in self._specs:
                    raise ValueError(
                        f"Capability {spec.id} hat unbekannte "
                        f"Abhängigkeit {parent}"
                    )

        state: dict[str, int] = {}

        def visit(name: str, path: list[str]) -> None:
            if state.get(name) == 1:
                cycle = " -> ".join([*path, name])
                raise ValueError(f"Zyklische Capability: {cycle}")

            if state.get(name) == 2:
                return

            state[name] = 1

            for parent in self._specs[name].parents:
                visit(parent, [*path, name])

            state[name] = 2

        for capability_id in self._specs:
            visit(capability_id, [])
