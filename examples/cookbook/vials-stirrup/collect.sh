#!/usr/bin/env bash
# VIALS with Stirrup: 161 questions, three samples each, in containers without network.
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL:?model ID served by the endpoint}"
: "${MODEL_BASE:?Chat Completions API base URL}"
: "${JUDGE_BASE_URL:?judge API base URL}"
: "${JUDGE_API_KEY:?judge API key}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-6-luna}"
export JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-medium}"

data="${VIALS_DATA:-vials-data}"
mkdir -p "$data"

roller collect --bench vials --bench-root "$data" --samples 3 \
  --model "$MODEL" --api-base "$MODEL_BASE" --harness stirrup \
  --docker-image "${VIALS_IMAGE:-roller-vials}" --network none \
  --thinking max --stirrup-max-steps 100 --stirrup-context-length 98304 \
  --stirrup-output-tokens 65536 --stirrup-command-timeout 300 \
  --stirrup-temperature 1 --stirrup-top-p 0.95 --stirrup-seed 42 \
  --stirrup-request-set n=1 --stirrup-request-set stream=false \
  --retries 0 --parallel "${PARALLEL:-4}" --grader-parallel "${GRADER_PARALLEL:-4}" \
  --keep-thinking --wire-trace full --capture-calls \
  --output "${OUTPUT_ROOT:-runs}" --run-name "${RUN_NAME:-vials-stirrup-$MODEL}" \
  "$@"
