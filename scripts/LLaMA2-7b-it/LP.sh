#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

TASKS="${TASKS:-STS12 STS13 STS14 STS15 STS16 STSBenchmark SICK-R}" \
PROMPT_SETTINGS="${PROMPT_SETTINGS:-language-specific}" \
SEMANTIC_VECTORS="${SEMANTIC_VECTORS:-mean}" \
LANGUAGE_COUNTS="${LANGUAGE_COUNTS:-6}" \
DIMENSION_SIZES="${DIMENSION_SIZES:-1024}" \
MODEL_PATH="${MODEL_PATH:-/path/to//llama-2-7b-chat}" \
MODEL_TAG="${MODEL_TAG:-meta-llama/Llama-2-7B-chat}" \
bash run_dydim_eval.sh
