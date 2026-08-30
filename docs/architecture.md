# Architecture

COGA separates reproducible source transformations from expensive model execution.

## Modules

- `coga.rst` downloads the two official RST repositories, checks every published shard hash,
  rejects unsafe TAR members, and extracts idempotently.
- `coga.cohorts` scans public task/trajectory JSON, applies the frozen RST prompt and policy
  normalization, and writes compact `population.sqlite` and physically separate
  `outcomes.sqlite` databases.
- `coga.data` builds same-task pairs, seven controls, ATIF turns, visible chat history, and
  target-only labels.
- `scripts/scoring/score_gradients.py` owns model warm-up, per-example gradients,
  CountSketch, cross-fit prototypes, and the statistical gate.
- `scripts/data/materialize_sft.py` performs deterministic cohort selection, conservative
  leakage filtering, round-robin sampling, and exact loss-token capping.
- `scripts/training/train_qlora.py` owns one independent training arm. It never loads the
  other arm's adapter.
- `scripts/evaluation/` owns external services and result validation.

## Data boundaries

```text
data/raw/          immutable Hub snapshots
data/extracted/    immutable extracted packages + extraction markers
data/processed/    compact deterministic cohort databases
runs/              gradients, selected tensors, adapters, and reports
```

Only `instruction + rollout-visible messages + same-step observations` enter SFT rendering.
Solutions, tests, verifier text, rewards, future observations, and COGA scores are excluded.
Outcome labels live in a separate SQLite file to make accidental access visible.

## Failure behavior

All critical stages fail closed: hash mismatch, unsafe archive path, unexpected cohort count,
changed gradient contract, failed gradient gate, unequal token budget, non-finite loss,
incomplete benchmark tasks, or unparsable official BFCL score stops the pipeline.
