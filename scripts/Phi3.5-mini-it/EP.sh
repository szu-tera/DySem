#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

TASKS="${TASKS:-STS12 STS13 STS14 STS15 STS16 STSBenchmark SICK-R}" \
PROMPT_SETTINGS="${PROMPT_SETTINGS:-english}" \
SEMANTIC_VECTORS="${SEMANTIC_VECTORS:-mean}" \
LANGUAGE_COUNTS="${LANGUAGE_COUNTS:-10}" \
DIMENSION_SIZES="${DIMENSION_SIZES:-1024}" \
MODEL_PATH="${MODEL_PATH:-/path/to//Phi-3.5-mini-instruct}" \
MODEL_TAG="${MODEL_TAG:-Phi/Phi-3.5-mini-instruct}" \
bash run_dydim_eval.sh
