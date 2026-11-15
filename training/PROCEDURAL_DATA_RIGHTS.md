# Review for the local procedural training corpus v1

Review date: 2026-11-02. Scope: data produced by
src/aion_u1/training/procedural.py, under the user's explicit AION training goal.

Sources: project-authored prompt templates, Python integer arithmetic, sorting,
fixed JSON structures and short project-authored arithmetic functions.
No third-party dataset, Teacher output, private conversation, personal record
or protected reasoning trace is included.

The generator checks each target using deterministic computations, JSON
structure checks or bounded tests of the generator's own fixed function template.
Training permission is recorded only for these internally constructed examples.
The permission is for this project's training work; it is not an approval of
the separately stored, unreviewed Teacher candidates.

License identifier: project-authored-procedural-v1. This identifier records
internal provenance and the authorized use. It does not assign a public license
to the user's project.

The train, development and held-out test sets use disjoint numerical ranges and
distinct prompts. They measure limited generalization to unseen parameters
within the listed procedural task families. They do not prove broad reasoning,
code quality, native tool execution, multilingual competence or injection safety.
The development split may select checkpoints; only the separate holdout is
reserved for the final before/after comparison.
