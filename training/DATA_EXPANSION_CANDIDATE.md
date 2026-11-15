# AION-U1: candidate data expansion v1

This candidate adds four project-authored procedural task families: subtraction,
multiplication, finding the larger integer in JSON, and filtering an integer list.
The generator creates 96 training and 24 development candidates using separate
parameter ranges and validates each target deterministically. No external sources,
model calls, personal data, or protected reasoning traces are used.

These rows are deliberately NOT approved for training: license/rights review and
quality review must be documented before setting training_approved. No claims of
independent review or actual improvement are made. The previous holdout and its
hash are untouched; exact prompt overlap with all three previous splits is checked
when the previous corpus is available. Exact-prompt separation does not guarantee
semantic independence. The 24 existing holdout cases only cover the original six
families, so they cannot establish quality on these four new families. A separate,
independently reviewed evaluation set is necessary before measuring generalization.

Generate into a fresh directory with:

    python scripts/generate_data_expansion_candidate.py data/generated/authored-expansion-candidate-v1

The generator refuses an existing output directory. `manifest.json` records the
generator hash and hashes of the two data files. It contains no holdout split.
