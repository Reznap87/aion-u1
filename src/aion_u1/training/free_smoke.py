"""Single synthetic free inference check; never produces training data."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from ..teachers.free_only import FreeOnlyClient, MODEL_PATTERN
from .run_ledger import RunLedger

PROMPT = "Respond with exactly AION_OK."
MAX_TOKENS = 32


async def smoke(client: FreeOnlyClient, model: str, ledger: RunLedger) -> dict:
    if not re.fullmatch(MODEL_PATTERN, model):
        raise ValueError("Explicit free model variant required")
    await client.require_zero_limit()
    if model not in await client.list_models():
        raise ValueError("Free model pricing could not be verified")
    reservation = ledger.reserve({"kind": "synthetic_free_smoke_v1",
                                  "model": model, "planned_candidates": 1})
    try:
        response = await client.generate(
            model, [{"role": "user", "content": PROMPT}],
            temperature=0.0, max_tokens=MAX_TOKENS,
        )
        passed = response.content.strip() == "AION_OK" and response.finish_reason == "stop"
    except BaseException:
        ledger.finish(reservation, success=False)
        raise
    ledger.finish(reservation, success=passed)
    return {"kind": "synthetic_free_smoke_v1", "model": model,
            "passed": passed, "training_approved": False,
            "reported_cost_verified_zero": True, "requests": 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--execute-free", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(MODEL_PATTERN, args.model):
        parser.error("Explicit free model variant required")
    if not args.execute_free:
        print(json.dumps({"model": args.model, "executed": False, "requests": 0}))
        return 0
    if args.ledger is None:
        parser.error("Persistent ledger required")
    key = os.getenv("AION_OPENROUTER_FREE_API_KEY", "")
    if not key:
        parser.error("Dedicated free-only API key missing")

    async def run():
        client = FreeOnlyClient(key)
        try:
            return await smoke(client, args.model, RunLedger(args.ledger))
        finally:
            await client.close()

    result = asyncio.run(run())
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
