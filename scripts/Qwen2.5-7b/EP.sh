#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

TASKS="${TASKS:-STS12 STS13 STS14 STS15 STS16 STSBenchmark SICK-R}" \
PROMPT_SETTINGS="${PROMPT_SETTINGS:-english}" \
SEMANTIC_VECTORS="${SEMANTIC_VECTORS:-mean}" \
LANGUAGE_COUNTS="${LANGUAGE_COUNTS:-12}" \
DIMENSION_SIZES="${DIMENSION_SIZES:-1024}" \
MODEL_PATH="${MODEL_PATH:-/path/to//qwen2.5-7B}" \
MODEL_TAG="${MODEL_TAG:-Qwen/Qwen2.5-7B}" \
bash run_dydim_eval.sh
