"""Measure the local Transformers loading path on a small synthetic model.

Requires psutil, torch, transformers, accelerate and safetensors already installed.
The generated model is random and carries no quality claims. No network is used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
from inference_memory_preflight import windows_memory

GIB = 1024 ** 3


def reserve_ok(minimum: int) -> bool:
    memory = windows_memory()
    return min(memory.values()) >= minimum


def process_tree_memory(process: psutil.Process, descendants: dict | None = None) -> tuple[int, int]:
    rss = private = 0
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    if descendants is not None:
        descendants.update((member.pid, member) for member in children)
    for member in [process, *children]:
        try:
            info = member.memory_info()
            rss += info.rss
            private += getattr(info, "private", 0)
        except psutil.NoSuchProcess:
            pass
    return rss, private


def stop_child(child, process: psutil.Process | None, descendants: dict) -> None:
    """Stop observed launcher children even if the launcher has already exited."""
    deadline = time.monotonic() + 10
    if process is not None:
        try:
            descendants.update((member.pid, member) for member in process.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    for member in reversed(list(descendants.values())):
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
    if child.poll() is None:
        try:
            child.kill()
        except ProcessLookupError:
            pass
    child.wait(timeout=max(0, deadline - time.monotonic()))
    if descendants:
        _, alive = psutil.wait_procs(list(descendants.values()),
                                     timeout=max(0, deadline - time.monotonic()))
        if alive:
            raise RuntimeError("Synthetic model child processes did not stop")


def child_make(directory: Path) -> None:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(42)
    config = GPT2Config(vocab_size=12000, n_positions=128, n_embd=384, n_layer=6, n_head=6)
    model = GPT2LMHeadModel(config).float()
    model.save_pretrained(directory, safe_serialization=True, max_shard_size="40MB")


def child_load(directory: Path, dtype: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("Missing parent start")
    model = AutoModelForCausalLM.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, low_cpu_mem_usage=True, torch_dtype=getattr(torch, dtype),
    )
    model.to("cpu").eval()
    with torch.inference_mode():
        tokens = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False,
                                pad_token_id=0)
    print(json.dumps({"actual_dtype": str(next(model.parameters()).dtype),
                      "output_token_count": int(tokens.shape[-1] - 3)}), flush=True)


def measure(directory: Path, dtype: str) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    child = subprocess.Popen([sys.executable, __file__, "--child-load", str(directory), dtype],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, env=env)
    process = None
    descendants = {}
    try:
        # Child writes READY after heavyweight imports. Bound every phase.
        started = time.monotonic()
        process = psutil.Process(child.pid)
        # stdout.readline is bounded by the child initialization watchdog below.
        import threading
        line = []
        waiter = threading.Thread(target=lambda: line.append(child.stdout.readline()), daemon=True)
        waiter.start()
        while waiter.is_alive():
            process_tree_memory(process, descendants)
            if time.monotonic() - started > 90 or not reserve_ok(GIB):
                raise RuntimeError("Initialization timed out or memory reserve fell below 1 GiB")
            waiter.join(0.05)
        if not line or line[0].strip() != "READY":
            raise RuntimeError("Child did not initialize")
        baseline_rss, baseline_private = process_tree_memory(process, descendants)
        peak_rss, peak_private = baseline_rss, baseline_private
        child.stdin.write("GO\n")
        child.stdin.flush()
        while child.poll() is None:
            if time.monotonic() - started > 180 or not reserve_ok(GIB):
                raise RuntimeError("Inference timed out or memory reserve fell below 1 GiB")
            try:
                rss, private = process_tree_memory(process, descendants)
                peak_rss = max(peak_rss, rss)
                peak_private = max(peak_private, private)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        # A surviving descendant can keep the pipe open after its launcher exits.
        output, _ = child.communicate(timeout=10)
        output = output.strip().splitlines()
        if child.returncode != 0 or not output:
            raise RuntimeError("Synthetic model load or generation failed")
        return {"requested_dtype": dtype, "baseline_rss_bytes": baseline_rss,
                "peak_rss_bytes": peak_rss, "baseline_private_bytes": baseline_private,
                "peak_private_bytes": peak_private, **json.loads(output[-1])}
    finally:
        stop_child(child, process, descendants)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child-make":
        child_make(Path(sys.argv[2]))
        return 0
    if len(sys.argv) == 4 and sys.argv[1] == "--child-load":
        child_load(Path(sys.argv[2]), sys.argv[3])
        return 0
    if len(sys.argv) != 1:
        raise ValueError("Unexpected arguments")
    if not reserve_ok(2 * GIB):
        raise RuntimeError("At least 2 GiB physical and commit memory must be available")
    with tempfile.TemporaryDirectory(prefix="aion-synthetic-load-") as folder:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        subprocess.run([sys.executable, __file__, "--child-make", folder], check=True, timeout=90, env=env)
        weights = sum(p.stat().st_size for p in Path(folder).glob("*.safetensors"))
        reports = [measure(Path(folder), dtype) for dtype in ("float32", "float16")]
        print(json.dumps({"synthetic_model": "random GPT-2", "weights_bytes": weights,
                          "results": reports, "real_aion_measured": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
