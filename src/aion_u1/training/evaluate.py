"""Evaluiert ein lokal geladenes Modell auf der Capability-Suite.

Der AION-Index ist sonst nur fuer Lehrer messbar; dieser Befehl misst das
ergebnis eines SFT-Laufs (Basismodell oder Basismodell mit LoRA-Adapter) mit
derselben Suite und derselben Aggregation. Damit wird GOAL.mds zweite Bedingung
(„keine Kernfaehigkeit unter 95 % des jeweiligen besten Lehrers") ueberhaupt
pruefbar.

Aufruf:  aion-u1-evaluate-model --base-model <id> [--adapter <pfad>] \
           --suite evaluations/capability_suite/smoke.yaml \
           --output reports/eval-model.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aion_u1 import __version__  # noqa: E402,F401
from aion_u1.audit import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    build_case_input,
    calculate_scores,
    verify_response,
    evaluation_metadata,
)
from aion_u1.capabilities import CapabilityRegistry  # noqa: E402
from aion_u1.cli import load_cases, repository_root  # noqa: E402
from aion_u1.models import CaseResult  # noqa: E402
from aion_u1.provenance import collect as collect_provenance  # noqa: E402
from aion_u1.training.model_identity import model_identity  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-u1-evaluate-model",
        description="Evaluiert ein lokales Modell auf der Capability-Suite.",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", default=None, help="Full Hub commit SHA for reproducible evaluation")
    parser.add_argument("--adapter", default=None, help="LoRA-Adapter-Verzeichnis")
    parser.add_argument(
        "--suite",
        default="evaluations/capability_suite/smoke.yaml",
    )
    parser.add_argument("--output", default="reports/eval-model.json")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="nur die ersten N Faelle (Schnelltest)",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    return parser


def _laden(base_model: str, adapter: str | None, device: str, revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lade_device = {"cpu": "cpu", "cuda": "cuda", "auto": None}[device]
    revision_kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model, **revision_kwargs)
    kwargs = {"low_cpu_mem_usage": True, **revision_kwargs}
    if lade_device:
        kwargs["device_map"] = lade_device
    if torch.cuda.is_available() and (device == "cuda" or device == "auto"):
        kwargs["torch_dtype"] = torch.float16
    modell = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter:
        from peft import PeftModel

        modell = PeftModel.from_pretrained(modell, adapter)
    modell.eval()
    return modell, tokenizer


def _ausfuehren(modell, tokenizer, case, max_tokens: int, root: Path) -> CaseResult:
    eingabe = build_case_input(case, root)
    if not isinstance(eingabe, str):
        return CaseResult(
            teacher_alias="lokal",
            teacher_model_id="lokal",
            case_id=case.id,
            capability=case.capability,
            prompt=case.prompt,
            response="",
            score=None,
            passed=False,
            verifier_type=case.verifier.type,
            error="Multimodale Faelle werden lokal noch nicht ausgefuehrt",
            latency_ms=0,
            max_tokens=max_tokens,
        )
    import torch

    token = tokenizer(eingabe, return_tensors="pt")
    token = {schluessel: wert.to(modell.device) for schluessel, wert in token.items()}
    start = time.time()
    with torch.no_grad():
        ausgabe = modell.generate(
            **token,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = int((time.time() - start) * 1000)
    text = tokenizer.decode(ausgabe[0][token["input_ids"].shape[1]:], skip_special_tokens=True)
    verification = verify_response(text, case.verifier)
    truncated = len(ausgabe[0]) - token["input_ids"].shape[1] >= max_tokens
    return CaseResult(
        teacher_alias="lokal",
        teacher_model_id="lokal",
        case_id=case.id,
        capability=case.capability,
        prompt=case.prompt,
        response=text,
        score=verification.score,
        passed=verification.passed and not truncated,
        error=verification.reason if verification.error else None,
        verifier_type=case.verifier.type,
        latency_ms=latency,
        max_tokens=max_tokens,
        finish_reason="length" if truncated else "stop",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from aion_u1.cli import repository_root as find_project_root

    root = find_project_root()
    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = root / suite_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    cases = load_cases(suite_path)
    if args.limit:
        cases = cases[: args.limit]
    registry = CapabilityRegistry.load(root / "configs" / "capabilities.yaml")
    for case in cases:
        registry.get(case.capability)

    print(f"Evaluiere {len(cases)} Faelle mit {args.base_model}"
          + (f" + Adapter {args.adapter}" if args.adapter else ""), file=sys.stderr)
    identity = model_identity(args.base_model, args.revision, args.adapter)
    modell, tokenizer = _laden(args.base_model, args.adapter, args.device, args.revision)

    ergebnisse = [_ausfuehren(modell, tokenizer, case, args.max_tokens, root) for case in cases]
    if model_identity(args.base_model, args.revision, args.adapter) != identity:
        raise RuntimeError("Model artifacts changed during evaluation")
    scores = calculate_scores(ergebnisse, registry)
    score = scores.get("lokal")

    report = {
        "schema_version": 1,
        **evaluation_metadata(cases),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "local_model_evaluation",
        "base_model": args.base_model,
        "model_identity": identity,
        "adapter": args.adapter,
        "suite": suite_path.stem,
        "suite_path": str(suite_path),
        "max_tokens": args.max_tokens,
        "results": [r.model_dump() for r in ergebnisse],
        "scores": {k: v.model_dump() for k, v in scores.items()},
        "provenance": collect_provenance(
            root,
            {"suite": suite_path, "capabilities": root / "configs" / "capabilities.yaml"},
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if score is not None:
        print(f"AION-Index (lokal): {score.overall_score}", file=sys.stderr)
        print(f"gemessen: {score.measured_score}  coverage: {score.coverage}", file=sys.stderr)
    else:
        print("Kein auswertbarer Fall — kein Index.", file=sys.stderr)
    print(f"Report: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
