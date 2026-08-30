# Full reproduction

## 1. Install

```bash
bash scripts/setup/install.sh
source .venv/bin/activate
bash scripts/setup/download_model.sh
```

For shared clusters, set `COGA_VENV` and `COGA_MODEL_DIR` before running the setup scripts.
Edit only path/executable fields in a copied config; preserve scientific parameters when
claiming a reproduction.

## 2. Prepare RST

```bash
bash scripts/data/prepare_rst.sh configs/qwen3_8b_4090.json
```

Allow roughly 28 GB for archives and 26 GB for extracted data. The cohort scan reads the
public JSON corpus once and may take hours on network storage. Completion is defined by a
verified download report, extraction report, and a 255-cohort report.

## 3. Score and curate

```bash
python scripts/data/build_manifests.py --config configs/qwen3_8b_4090.json
python scripts/scoring/score_gradients.py --config configs/qwen3_8b_4090.json
python scripts/data/materialize_sft.py --config configs/qwen3_8b_4090.json
```

The gradient command can be repeated after interruption; complete row caches are reused only
when the frozen contract matches. Inspect `gradient_report.json`. Data materialization is
supposed to stop when `method_gate != PASS`.

## 4. Train

```bash
bash scripts/training/train.sh configs/qwen3_8b_4090.json
```

The two arms run sequentially on one GPU. Preserve `completion_report.json`, adapter files,
hashes, peak memory, elapsed time, and complete logs for reporting.

## 5. Evaluate

Follow the official Terminal-Bench/Harbor and BFCL links in the README. Export all 89
Terminal-Bench 2.1 tasks locally, freeze the tree, then run:

```bash
python scripts/evaluation/freeze_terminal_bench.py --config configs/qwen3_8b_4090.json
bash scripts/evaluation/evaluate.sh configs/qwen3_8b_4090.json
```

The frozen BFCL scope is the official V4 Multi-Turn group. Never commit credentials,
downloaded task data, model weights, or logs.

## 6. Report

Report the gradient gate and CI, selected cohorts/rows/tokens, both training runtimes and peak
VRAM, all three Terminal-Bench success rates with paired CI, and all three official BFCL
overall scores. Do not substitute warm-up loss, sketch norms, smoke results, or numbers from
related papers for downstream results.
