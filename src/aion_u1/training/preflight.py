"""Offline SFT readiness report. Does not download models or start training."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..capabilities import CapabilityRegistry
from ..cli import load_cases
from ..privacy import scan_text
from ..provenance import sha256_file
from .config import SFTConfig, find_project_root, load_sft_config
from .data import is_training_approved, read_jsonl
from .model_identity import model_identity


def _key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def _inspect(rows: list[dict[str, Any]], known: set[str], benchmark_prompts: set[str]) -> dict:
    issues: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    ids, groups, prompts = set(), set(), set()
    for row in rows:
        identifier, group = row.get("id"), row.get("group_id")
        if not isinstance(identifier, str) or not identifier:
            issues["missing_id"] += 1
        elif identifier in ids:
            issues["duplicate_id"] += 1
        else:
            ids.add(identifier)
        if not isinstance(group, str) or not group:
            issues["missing_group"] += 1
        else:
            groups.add(group)
        if (row.get("verification") or {}).get("status") != "verified":
            issues["not_verified"] += 1
        if not is_training_approved(row):
            issues["not_approved"] += 1
        messages = row.get("messages")
        if (not isinstance(messages, list) or len(messages) < 2
                or any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                       or not m["content"].strip() for m in messages)
                or messages[-1].get("role") != "assistant"
                or not any(m.get("role") == "user" for m in messages[:-1])):
            issues["invalid_messages"] += 1
        else:
            if any(scan_text(m["content"]) for m in messages):
                issues["privacy_pattern"] += 1
            prompt = _key("\n".join(m["content"] for m in messages[:-1] if m.get("role") == "user"))
            if prompt in prompts:
                issues["duplicate_prompt"] += 1
            prompts.add(prompt)
            if prompt in benchmark_prompts:
                issues["benchmark_overlap"] += 1
        capabilities = row.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities or any(
            not isinstance(c, str) or c not in known for c in capabilities
        ):
            issues["invalid_capabilities"] += 1
        else:
            counts.update(set(capabilities))
    return {"rows": len(rows), "issues": dict(issues), "capability_counts": dict(counts),
            "ids": ids, "groups": groups, "prompts": prompts}


def check(config: SFTConfig, root: Path, suite: Path, capability_path: Path) -> dict:
    train_path = root / config.data.train_path
    eval_path = root / config.data.eval_path if config.data.eval_path else None
    paths = {"train": train_path, "eval": eval_path, "suite": suite, "capabilities": capability_path}
    hashes = {name: sha256_file(path) if path else None for name, path in paths.items()}
    if any(hashes[name] is None for name in ("train", "suite", "capabilities")) or (eval_path and hashes["eval"] is None):
        raise ValueError("Preflight input is unreadable")
    registry = CapabilityRegistry.load(capability_path)
    known = {spec.id for spec in registry.list()}
    cases = load_cases(suite)
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Benchmark suite is empty or has duplicate case IDs")
    for case in cases:
        registry.get(case.capability)
    benchmark = {_key(case.prompt) for case in cases}
    train = _inspect(read_jsonl(train_path), known, benchmark)
    evaluation = _inspect(read_jsonl(eval_path) if eval_path else [], known, benchmark)
    blockers = []
    for label, split in (("train", train), ("eval", evaluation)):
        if not split["rows"]:
            blockers.append(f"{label}:empty")
        blockers.extend(f"{label}:{issue}" for issue in sorted(split["issues"]))
    overlaps = {field: len(train[field] & evaluation[field]) for field in ("ids", "groups", "prompts")}
    blockers.extend(f"split_overlap:{field}" for field, count in overlaps.items() if count)
    if not config.data.require_training_approval:
        blockers.append("training_approval_required")
    if not config.model.license_acknowledged or config.model.trust_remote_code:
        blockers.append("model_license_or_remote_code")
    identity = None
    base = Path(config.model.base_model)
    if base.is_dir():
        identity = model_identity(str(base), None, None)
    elif re.fullmatch(r"[0-9a-f]{40}", config.model.revision):
        identity = model_identity(config.model.base_model, config.model.revision, None)
    else:
        blockers.append("immutable_model_revision_required")
    output = root / config.run.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        blockers.append("output_not_empty")
    if config.run.overwrite_output_dir or config.run.resume_from_checkpoint:
        blockers.append("fresh_run_required")
    if hashes != {name: sha256_file(path) if path else None for name, path in paths.items()}:
        raise RuntimeError("Preflight inputs changed during validation")
    for split in (train, evaluation):
        for field in ("ids", "groups", "prompts"):
            del split[field]
    return {
        "schema_version": 1, "kind": "offline_sft_preflight",
        "ready": not blockers, "blockers": blockers, "model_identity": identity,
        "train": train, "eval": evaluation, "split_overlap": overlaps,
        "uncovered_train_capabilities": sorted(known - set(train["capability_counts"])),
        "uncovered_eval_capabilities": sorted(known - set(evaluation["capability_counts"])),
        "hashes": hashes,
        "training_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "suite", "capabilities"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_sft_config(args.config)
    report = check(config, find_project_root(), args.suite, args.capabilities)
    report["hashes"]["config"] = sha256_file(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
