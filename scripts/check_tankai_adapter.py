"""Synthetic transport check using the real adapter from a TankAICore checkout.

No trained model, Teacher request, external inference or training is involved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import subprocess
import threading
from pathlib import Path

from openai import AuthenticationError, NotFoundError

from aion_u1.inference import Completion
from aion_u1.server import InferenceServer


class SyntheticEngine:
    model_id = "aion-u1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> Completion:
        assert request.model == self.model_id
        assert [message.role for message in request.messages] == ["system", "user"]
        self.calls += 1
        return Completion("synthetic adapter result", 6, 3, "stop")


def commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def check(tankai_root: Path) -> dict:
    adapter_path = tankai_root / "tankai" / "core" / "llm.py"
    spec = importlib.util.spec_from_file_location("tankai_adapter_contract", adapter_path)
    if spec is None or spec.loader is None:
        raise ValueError("TankAI-Adapter kann nicht geladen werden.")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    engine = SyntheticEngine()
    key = secrets.token_urlsafe(32)
    checks = {}
    with InferenceServer(("127.0.0.1", 0), engine, key) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        clients = []
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}/v1"
            def client(model="aion-u1", api_key=key):
                llm = adapter.OpenAILLM(
                    model=model, api_key=api_key, base_url=base,
                    max_tokens=8, timeout_seconds=3, max_retries=0,
                )
                clients.append(llm.client)
                return llm
            budget = adapter.LLMCallBudget(1)
            llm = adapter.BudgetedLLM(client(), budget)
            result = llm.complete("Synthetic user prompt.", system="Synthetic system prompt.")
            checks["core_adapter_completion"] = result == "synthetic adapter result"
            checks["system_and_user_messages"] = engine.calls == 1
            try:
                llm.complete("Blocked second call.", system="Synthetic system prompt.")
            except RuntimeError:
                checks["core_call_budget_blocks_second_request"] = engine.calls == 1
            else:
                checks["core_call_budget_blocks_second_request"] = False
            try:
                client(api_key="synthetic-wrong-key").complete("Synthetic auth test.")
            except AuthenticationError:
                checks["authentication_failure_reaches_core"] = True
            else:
                checks["authentication_failure_reaches_core"] = False
            try:
                client(model="unavailable-model").complete("Synthetic model test.")
            except NotFoundError:
                checks["unknown_model_has_no_fallback"] = engine.calls == 1
            else:
                checks["unknown_model_has_no_fallback"] = False
        finally:
            for connection in clients:
                connection.close()
            server.shutdown()
            thread.join(timeout=3)
    if not all(checks.values()):
        raise RuntimeError("TankAI-Adaptervertrag nicht vollständig bestanden.")
    return {
        "schema_version": "1.0",
        "kind": "synthetic_tankai_adapter_check",
        "tankai_commit": commit(tankai_root),
        "aion_commit": commit(Path(__file__).resolve().parents[1]),
        "checks": checks,
        "synthetic_model_calls": engine.calls,
        "external_model_calls": 0,
        "trained_model_verified": False,
        "quality_verified": False,
        "production_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tankai-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # This check is strictly loopback, including in environments with proxies.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    receipt = check(args.tankai_root.expanduser().resolve())
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
