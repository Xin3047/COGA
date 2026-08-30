#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION="b968826d9c46dd6066d109eabc6255188de91218"
DESTINATION="${COGA_MODEL_DIR:-${ROOT_DIR}/models/Qwen3-8B}"

hf download Qwen/Qwen3-8B \
  --revision "${REVISION}" \
  --local-dir "${DESTINATION}"
echo "Qwen3-8B ready: ${DESTINATION}"
