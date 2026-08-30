#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/configs/qwen3_8b_4090.json}"

cd "${ROOT_DIR}"
python -m coga.cli_data all --config "${CONFIG}"
