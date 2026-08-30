#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${1:-${ROOT_DIR}/configs/qwen3_8b_4090.json}"

cd "${ROOT_DIR}"
"${PYTHON}" scripts/evaluation/terminal_bench.py --config "${CONFIG}"
"${PYTHON}" scripts/evaluation/summarize_terminal_bench.py --config "${CONFIG}"
"${PYTHON}" scripts/evaluation/bfcl.py --config "${CONFIG}"
