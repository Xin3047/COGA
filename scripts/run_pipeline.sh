#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${1:-${ROOT_DIR}/configs/qwen3_8b_4090.json}"

cd "${ROOT_DIR}"
"${PYTHON}" -m coga.cli_data all --config "${CONFIG}"
"${PYTHON}" scripts/data/build_manifests.py --config "${CONFIG}"
"${PYTHON}" scripts/scoring/score_gradients.py --config "${CONFIG}"
"${PYTHON}" scripts/data/materialize_sft.py --config "${CONFIG}"
PYTHON="${PYTHON}" bash scripts/training/train.sh "${CONFIG}"

if [[ "${COGA_RUN_EVALUATION:-0}" == "1" ]]; then
  PYTHON="${PYTHON}" bash scripts/evaluation/evaluate.sh "${CONFIG}"
fi
