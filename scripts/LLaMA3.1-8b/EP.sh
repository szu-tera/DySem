#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

TASKS="${TASKS:-STS12 STS13 STS14 STS15 STS16 STSBenchmark SICK-R}" \
PROMPT_SETTINGS="${PROMPT_SETTINGS:-english}" \
SEMANTIC_VECTORS="${SEMANTIC_VECTORS:-source}" \
LANGUAGE_COUNTS="${LANGUAGE_COUNTS:-11}" \
DIMENSION_SIZES="${DIMENSION_SIZES:-1024}" \
MODEL_PATH="${MODEL_PATH:-/path/to//llama-3.1-8B}" \
MODEL_TAG="${MODEL_TAG:-meta-llama/Llama-3.1-8B}" \
bash run_dydim_eval.sh
