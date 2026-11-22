"""Generate an unapproved, reproducible expansion; never overwrite existing corpora."""
import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = ("subtract", "multiply", "compare", "range_filter")


def make_case(family, i, split):
    # Split ranges and distinct prompt wording reduce accidental overlap.
    a = (10 if split == "train" else 110) + i
    b = 2 + (i % 7)
    if family == "subtract":
        return f"Berechne {a} minus {b}. Gib ausschließlich die ganze Zahl aus.", str(a - b)
    if family == "multiply":
        return f"Berechne {a} mal {b}. Gib ausschließlich die ganze Zahl aus.", str(a * b)
    if family == "compare":
        return f'Vergleiche {a} und {a + b}. Gib nur JSON mit dem Feld "groesser" und der größeren Zahl aus.', json.dumps({"groesser": a + b}, ensure_ascii=False, separators=(",", ":"))
    if family == "range_filter":
        values = [a - b, a, a + b, a + 2 * b]
        return f'Wähle aus {json.dumps(values)} alle Zahlen größer als {a}. Antworte nur als JSON-Array in derselben Reihenfolge.', json.dumps([x for x in values if x > a], separators=(",", ":"))
    raise ValueError(family)


def verify(family, i, split, answer):
    a = (10 if split == "train" else 110) + i
    b = 2 + i % 7
    try:
        if family == "subtract": return int(answer) + b == a and str(int(answer)) == answer
        if family == "multiply": return int(answer) // b == a and int(answer) % b == 0 and str(int(answer)) == answer
        if family == "compare": return json.loads(answer) == {"groesser": max(a, a + b)}
        if family == "range_filter": return json.loads(answer) == [a + b, a + 2 * b]
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    return False


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"corpus": "aion-authored-expansion-candidate-v1", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "project-authored deterministic templates", "license": "project-authored; rights review pending", "training_approved": False, "external_calls": 0, "limitations": "Synthetic narrow tasks; no broad-quality or independent benchmark claim", "splits": {}}
    prompts = set()
    for split, count in (("train", 24), ("dev", 6)):
        rows = []
        for family in FAMILIES:
            for i in range(count):
                prompt, answer = make_case(family, i, split)
                if prompt in prompts or not verify(family, i, split, answer): raise ValueError("overlap or invalid target")
                prompts.add(prompt)
                identity = f"{split}:{family}:{i}"
                rows.append({"id": "candidate-" + hashlib.sha256(identity.encode()).hexdigest()[:24], "group_id": identity, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "capabilities": ["reasoning.logical" if family in ("subtract", "multiply") else "instruction.structured_output"], "provenance": {"source": "project-authored-expansion-candidate-v1", "license": "pending", "training_approved": False}, "verification": {"status": "verified", "method": "deterministic-" + family}, "evaluation": {"family": family, "split": split}})
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        (output / (split + ".jsonl")).write_bytes(encoded)
        manifest["splits"][split] = {"rows": len(rows), "sha256": hashlib.sha256(encoded).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
