"""Conservative Windows CPU float16 preflight; this script never loads a model."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import struct
from pathlib import Path

GIB = 1024 ** 3
MAX_HEADER = 100 * 1024 * 1024


def model_budget(directory: Path) -> dict:
    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = index["weight_map"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Empty tensor index")
    by_shard = {}
    for name, shard in mapping.items():
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("Invalid shard name")
        by_shard.setdefault(shard, set()).add(name)
    target_bytes = 0
    max_source_bytes = 0
    for shard, expected in by_shard.items():
        path = directory / shard
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked shard")
        size = path.stat().st_size
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Invalid shard")
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= min(MAX_HEADER, size - 8):
                raise ValueError("Invalid safetensors header length")
            header = json.loads(f.read(length))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if set(tensors) != expected:
            raise ValueError("Shard and index differ")
        for tensor in tensors.values():
            if tensor.get("dtype") not in ("F32", "F16", "BF16"):
                raise ValueError("Unsupported tensor dtype")
            shape = tensor.get("shape")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("Invalid tensor shape")
            target_bytes += math.prod(shape) * 2
        max_source_bytes = max(max_source_bytes, size)
    # Conservative bound, not a measured peak: simultaneous float16 weights,
    # largest source shard, runtime overhead, plus spare capacity for the PC.
    required = target_bytes + max_source_bytes + 2 * GIB
    return {"float16_weight_bytes": target_bytes, "largest_source_shard_bytes": max_source_bytes,
            "runtime_allowance_bytes": GIB, "system_reserve_bytes": GIB,
            "required_free_bytes": required}


def windows_memory() -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows memory API required")
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page",
             "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    status = Memory()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {"available_physical_bytes": status.avail_phys, "available_commit_bytes": status.avail_page}


def assess(budget: dict, memory: dict) -> dict:
    required = budget["required_free_bytes"]
    return {**budget, **memory, "eligible": min(memory.values()) >= required,
            "basis": "conservative estimate; actual peak unmeasured"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    result = assess(model_budget(args.model_dir), windows_memory())
    print(json.dumps(result, indent=2))
    return 0 if result["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
