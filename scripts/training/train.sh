#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${1:-${ROOT_DIR}/configs/qwen3_8b_4090.json}"

cd "${ROOT_DIR}"
"${PYTHON}" scripts/training/train_qlora.py --config "${CONFIG}" --arm rejection_success
"${PYTHON}" scripts/training/train_qlora.py --config "${CONFIG}" --arm coga_selected
