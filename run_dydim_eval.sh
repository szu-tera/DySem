#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export DYDIM_ROOT="${DYDIM_ROOT:-${SCRIPT_DIR}}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export PROMPT_SETTINGS="${PROMPT_SETTINGS:-english language-specific}"
export SEMANTIC_VECTORS="${SEMANTIC_VECTORS:-source mean}"
export LANGUAGE_COUNTS="${LANGUAGE_COUNTS:-12}"
export DIMENSION_SIZES="${DIMENSION_SIZES:-256 512 768 1024 1280 2048}"
export TASKS="${TASKS:-STS12 STS13 STS14 STS15 STS16 STS17 STSBenchmark SICK-R}"
export CANDIDATE_LANGUAGES="${CANDIDATE_LANGUAGES:-eng_Latn zho_Hans fra_Latn deu_Latn spa_Latn rus_Cyrl arb_Arab jpn_Jpan kor_Hang ita_Latn por_Latn hin_Deva}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export TRANSLATION_MODEL="${TRANSLATION_MODEL:-facebook/nllb-200-distilled-600M}"
export TRANSLATION_DEVICE="${TRANSLATION_DEVICE:-cuda}"
export TRANSLATION_BATCH_SIZE="${TRANSLATION_BATCH_SIZE:-512}"
export TRANSLATION_MAX_LENGTH="${TRANSLATION_MAX_LENGTH:-256}"
export TRANSLATION_CACHE_NAMESPACE="${TRANSLATION_CACHE_NAMESPACE:-nllb-200-distilled-600M}"
export TRANSLATION_CACHE_DIR="${TRANSLATION_CACHE_DIR:-${SCRIPT_DIR}/translation_cache}"
export RANK_CACHE_DIR="${RANK_CACHE_DIR:-${SCRIPT_DIR}/rank_cache}"
export RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
export MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/configs/models.yaml}"
export FORCE_RERANK="${FORCE_RERANK:-0}"

echo "DyDim root: ${DYDIM_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Prompt settings: ${PROMPT_SETTINGS}"
echo "Semantic vectors: ${SEMANTIC_VECTORS}"
echo "Language counts: ${LANGUAGE_COUNTS}"
echo "Dimension sizes: ${DIMENSION_SIZES}"
echo "Tasks: ${TASKS}"
echo "Candidate languages: ${CANDIDATE_LANGUAGES}"
echo "Translation model: ${TRANSLATION_MODEL}"
echo "Translation device: ${TRANSLATION_DEVICE}"
echo "Translation batch size: ${TRANSLATION_BATCH_SIZE}"
echo "Translation cache: ${TRANSLATION_CACHE_DIR}"
echo "Rank cache: ${RANK_CACHE_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "Model config: ${MODEL_CONFIG}"
if [[ -n "${MODEL_SPECS:-}" ]]; then
  echo "Model specs: ${MODEL_SPECS}"
elif [[ -n "${MODEL_PATH:-}" || -n "${MODEL_TAG:-}" ]]; then
  echo "Model path: ${MODEL_PATH:-<unset>}"
  echo "Model tag: ${MODEL_TAG:-<unset>}"
fi

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" -m dydim
