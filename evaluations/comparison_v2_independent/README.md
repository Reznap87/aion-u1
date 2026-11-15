# Independent comparison evaluation v1

This small, manually authored set evaluates the comparison-v2 task with 18 cases:
six where the first value is greater, six where the second is greater, and six
where they are equal. It uses positive, negative, and mixed-sign integer pairs and
three prompt formulations distinct from the training generator. The prompts,
number pairs and targets were authored locally using basic arithmetic and the
Python standard library; no external dataset, teacher output, personal data or
model inference was used. The exact JSONL bytes are pinned in `manifest.json`.

This is an evaluation-only dataset. Its rows have `training_approved: false`;
keep it out of training and development/checkpoint selection. It does not replace
or alter the existing procedural-v1 holdout. `scorer.py` checks the requested
JSON object and integer using the signed difference of the given values. The
independent tests extract the two operands from the actual prompts and verify
metadata, targets, separation and failure of first/second position heuristics.

The score describes only this small synthetic comparison and formatting task.
It does not establish general reasoning, independent human review, model quality,
or actual model performance. The existing local evaluator is fixed to a different
holdout; it does not automatically run this set.

## Offline scoring of saved answers

Provide a UTF-8 JSONL file containing exactly one row for each evaluation ID.
Each row contains only `id` and `response` strings; `response` is the model's
actual answer text, not a JSON object embedded as a value. Example format:

    {"id":"comparison-independent-v1-01","response":"{\"groesser\":204}"}

Run from the project root, choosing a report path that does not exist:

    python evaluations/comparison_v2_independent/score_answers.py --answers path/to/answers.jsonl --output path/to/new-report.json

All 18 IDs must be present exactly once; unknown IDs, duplicate IDs, duplicate
JSON keys and missing rows cause failure before report creation. The command
checks the pinned evaluation hash, records both evaluation and input hashes,
per-case pass/fail and aggregate score, and refuses to overwrite a report.
It makes no model calls and cannot prove where saved answers originated; keep
model identity, generation settings and training-data isolation separately as
part of a future evaluation receipt. Do not use this reserved set to train or
select a checkpoint. This offline scorer does not alter the existing local
model evaluator or its separate procedural-v1 holdout.
