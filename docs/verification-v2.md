# Verification v2

Legacy suites retain their original text verification. Explicitly set
`engine: v2` for strict JSON extraction or partial contains_all scoring.
Partial scoring is opt-in and never applies to forbidden-content checks.

The experimental suite is `evaluations/capability_suite/benchmark_hard_v2.yaml`.
Its two code tasks use `python_function`, executing the candidate in isolated
Docker containers and comparing JSON return values outside the container.

## Setup and offline tests

Use a trusted Python image and record its resolved digest:

```sh
docker pull python:3.12-slim
export AION_VERIFIER_DOCKER_IMAGE="$(docker image inspect --format '{{index .RepoDigests 0}}' python:3.12-slim)"
pytest -q
pytest -q tests/test_functional_docker_live.py
```

The verifier itself never pulls an image. Missing or invalid image settings
fail closed. No model API is needed for these tests.

## Comparing evaluations

Teacher and student reports carry `index_version` and `verifier_image`.
The geometric aggregation formula is unchanged. Compare reports only when
the suite and capability hashes, index version, verifier image, and actual
case sets match. Check coverage, errors and truncation separately. The
existing student runner cannot evaluate image inputs, so a full index may
remain unavailable. A v2 label alone does not imply comparable results.

## Limitations

The functional verifier supports JSON-compatible function arguments and
return values, not arbitrary unittest scripts. Its fixed public cases can
be overfit; passing them is not proof of general coding competence.
Docker isolation is not an absolute security boundary. Use dedicated
workers for untrusted code. The older training unittest path still trusts
the process exit status and is not equivalent to the new verifier.

No new data is approved for training by these changes.
