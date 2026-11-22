"""Independently versioned project-authored integer-comparison data."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "training/COMPARISON_V2_RIGHTS.md"


def pair(index, split):
    if split not in ("train", "dev"):
        raise ValueError("unknown split")
    # Non-overlapping magnitudes; half of each relation uses negative values.
    magnitude = (10 if split == "train" else 110) + index // 6
    base = magnitude if index % 2 else -magnitude
    delta = 1 + index // 6 % 4
    relation = index % 6 // 2
    if relation == 0:
        return base + delta, base
    if relation == 1:
        return base, base + delta
    return base, base


def build(output):
    review = REVIEW.read_bytes()
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    review_hash = hashlib.sha256(review).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"corpus": "project-authored-comparison-v2", "generator_sha256": generator_hash,
        "review": "training/COMPARISON_V2_RIGHTS.md", "review_sha256": review_hash,
        "external_calls": 0, "training_approved": True, "splits": {}}
    seen = set()
    for split, count in (("train", 36), ("dev", 12)):
        rows = []
        for i in range(count):
            left, right = pair(i, split)
            prompt = f'Vergleiche {left} und {right}. Gib nur JSON mit dem Feld "groesser" und dem größeren Wert aus. Bei Gleichheit gib diesen gemeinsamen Wert aus.'
            if prompt in seen: raise ValueError("duplicate prompt")
            seen.add(prompt)
            target = left if left >= right else right
            answer = json.dumps({"groesser": target}, separators=(",", ":"))
            if json.loads(answer) != {"groesser": max(left, right)}: raise ValueError("invalid target")
            identity = f"v2:{split}:{left}:{right}"
            rows.append({"id": "compare-v2-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                "group_id": identity, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                "capabilities": ["instruction.structured_output"],
                "provenance": {"source": "project-authored-comparison-v2", "license": "project-authored-comparison-v2", "training_approved": True,
                    "approval": {"method": "project-authored-source-and-deterministic-verifier-review", "review": "training/COMPARISON_V2_RIGHTS.md", "review_sha256": review_hash, "authorization": "user authorized AION-U1 training-data development and training"}},
                "verification": {"status": "verified", "score": 1.0, "method": "integer-comparison-v2"},
                "evaluation": {"family": "compare-v2", "split": split}})
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        (output / f"{split}.jsonl").write_bytes(content)
        manifest["splits"][split] = {"rows": len(rows), "sha256": hashlib.sha256(content).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False, indent=2))
