"""Evaluate the pinned baseline or trained pilot on the reserved local holdout."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..inference import (
    ChatRequest, LocalModelEngine, ModelArtifact, load_model_artifact, read_json_object,
)
from .config import find_project_root, load_sft_config
from .data import read_jsonl
from .procedural import verify
from .sft import _sha256


def evaluate(config_path: Path, mode: str, output: Path) -> dict:
    import torch

    root = find_project_root()
    config = load_sft_config(config_path)
    holdout = root / "data/verified/procedural-v1/holdout.jsonl"
    manifest = read_json_object(holdout.parent / "data_manifest.json")
    if _sha256(holdout) != manifest["splits"]["holdout"]["sha256"]:
        raise ValueError("Holdout hash mismatch")
    if holdout.resolve() in {
        (root / config.data.train_path).resolve(),
        (root / config.data.eval_path).resolve() if config.data.eval_path else None,
    }:
        raise ValueError("Holdout is configured for training or development selection")
    rows = read_jsonl(holdout)
    if not rows or any(
        row.get("evaluation", {}).get("split") != "holdout"
        or row.get("provenance", {}).get("training_approved") is not False
        for row in rows
    ):
        raise ValueError("Reserved holdout required")
    if output.exists():
        raise FileExistsError("Evaluation output already exists")
    torch.set_num_threads(4)
    if mode == "base":
        directory = (root / config.model.base_model).resolve()
        source = read_json_object(directory / "source_revision.json")
        if source["revision"] != config.model.revision:
            raise ValueError("Baseline revision mismatch")
        digest = _sha256(directory / "model.safetensors")
        if digest != source["model_sha256"]:
            raise ValueError("Baseline weights changed")
        artifact = ModelArtifact(directory, digest, "not-applicable-base", source["revision"])
    else:
        run_dir = root / config.run.output_dir
        run = read_json_object(run_dir / "run_manifest.json")
        proof = run.get("training_proof", {})
        if (
            proof.get("optimizer_steps_this_invocation", 0) < 1
            or proof.get("changed_trainable_tensors", 0) < 1
        ):
            raise ValueError("No actual training proof for the student")
        artifact = load_model_artifact(run_dir)
    engine = LocalModelEngine(
        artifact, model_id=f"aion-u1-{mode}", context_window=128, dtype="float32"
    )
    cases = []
    families = defaultdict(lambda: {"passed": 0, "cases": 0})
    for row in rows:
        task = row["evaluation"]
        completion = engine.complete(ChatRequest(
            model=engine.model_id, messages=row["messages"][:-1],
            temperature=0.0, max_tokens=32,
        ))
        response = completion.content.strip()
        try:
            passed = verify(task["family"], task["a"], task["b"], response)
        except (ValueError, TypeError, KeyError, SyntaxError):
            passed = False
        families[task["family"]]["passed"] += int(passed)
        families[task["family"]]["cases"] += 1
        cases.append({
            "id": row["id"], "family": task["family"], "passed": bool(passed),
            "response": response, "expected": row["messages"][-1]["content"],
            "finish_reason": completion.finish_reason,
        })
        print(json.dumps({"mode": mode, "evaluated": len(cases), "total": len(rows)}), flush=True)
    report = {
        "kind": "procedural_holdout_evaluation",
        "mode": mode, "base_revision": artifact.base_revision,
        "artifact_sha256": artifact.sha256,
        "holdout_sha256": _sha256(holdout),
        "cases": len(cases), "passed": sum(case["passed"] for case in cases),
        "families": dict(families), "results": cases,
        "external_teacher_calls": 0, "frontier_quality_verified": False,
        "limitations": "Disjoint parameters within authored task families only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/sft_cpu_procedural.yaml"))
    parser.add_argument("--mode", choices=("base", "student"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.config, args.mode, args.output)
    print(json.dumps({key: report[key] for key in ("mode", "cases", "passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
