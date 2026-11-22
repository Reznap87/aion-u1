"""Score a complete set of saved answers against the pinned comparison evaluation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
EXPECTED_SHA256 = "abcb71d1bb5e5e089c50748f69401bfe511c143e3bf9e3a594bb34a4ebe7099e"
_spec = importlib.util.spec_from_file_location("comparison_eval_scorer", DIRECTORY / "scorer.py")
_scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer)


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _jsonl(content):
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"empty line {line_number}")
        row = json.loads(line, object_pairs_hook=_object_without_duplicates)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} must be an object")
        rows.append(row)
    return rows


def evaluate(answers_path, output_path):
    case_bytes = (DIRECTORY / "holdout.jsonl").read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != EXPECTED_SHA256:
        raise ValueError("pinned evaluation hash mismatch")
    manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_SHA256 or manifest.get("cases") != 18:
        raise ValueError("evaluation manifest mismatch")
    cases = _jsonl(case_bytes)
    if len(cases) != 18 or len({case["id"] for case in cases}) != 18:
        raise ValueError("evaluation cases invalid")
    for case in cases:
        if case["provenance"].get("training_approved") is not False or case["evaluation"].get("split") != "holdout":
            raise ValueError("reserved evaluation required")
    answer_bytes = Path(answers_path).read_bytes()
    supplied = {}
    for row in _jsonl(answer_bytes):
        if set(row) != {"id", "response"} or not isinstance(row["id"], str) or not isinstance(row["response"], str):
            raise ValueError("each answer must contain only string id and response")
        if row["id"] in supplied:
            raise ValueError("duplicate answer ID")
        supplied[row["id"]] = row["response"]
    expected_ids = {case["id"] for case in cases}
    if set(supplied) != expected_ids:
        raise ValueError("answer IDs must match every evaluation case exactly")
    results = []
    for case in cases:
        task = case["evaluation"]
        response = supplied[case["id"]]
        results.append({"id": case["id"], "passed": _scorer.score(task["left"], task["right"], response)})
    report = {"kind": "offline_saved_answer_comparison_evaluation", "cases": len(results),
        "passed": sum(row["passed"] for row in results), "evaluation_sha256": EXPECTED_SHA256,
        "answers_sha256": hashlib.sha256(answer_bytes).hexdigest(), "results": results,
        "model_execution_verified": False, "model_identity_verified": False,
        "limitations": "Only saved answers on 18 synthetic comparison cases; no model or general quality verified."}
    with Path(output_path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True, help="JSONL: each row has id and response strings")
    parser.add_argument("--output", type=Path, required=True, help="new report file (never overwritten)")
    args = parser.parse_args()
    report = evaluate(args.answers, args.output)
    print(json.dumps({"cases": report["cases"], "passed": report["passed"]}))


if __name__ == "__main__":
    main()
