# Review: project-authored comparison data v2

Scope: `scripts/generate_comparison_v2.py` and the exact generator hash recorded in
its data manifest. The prompts, integer pairs and JSON targets are constructed by
project code from arithmetic and Python standard-library functions. No external
datasets, model outputs, private records, or protected reasoning are used. Python
standard-library results here are facts, not copied third-party text.

The user explicitly authorized AION-U1 training-data development and training.
For this internally authored comparison corpus, that authorization supports use
in this project's training. The provenance identifier `project-authored-comparison-v2`
records the source and intended project use; it is not a public license grant or
permission for any other corpus. The old holdout is not modified or included.

Each split contains first-greater, second-greater, and equal examples, including
negative inputs. Disjoint numerical ranges and identities separate train and dev.
The data remain synthetic and the deterministic checker does not prove a model
learns the task. A separately designed evaluation is needed for a new quality claim.
