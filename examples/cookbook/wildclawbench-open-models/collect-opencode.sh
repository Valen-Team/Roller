#!/usr/bin/env bash
# Run one model exposed through streaming Chat Completions.
set -euo pipefail
cd "$(dirname "$0")"
: "${WILDCLAWBENCH_ROOT:?path of the WildClawBench checkout}"
: "${MODEL:?model ID served by the endpoint}"
: "${MODEL_BASE:?Chat Completions API base URL, including /v1}"
: "${MODEL_CONTEXT_WINDOW:?context window configured by the model service}"
: "${OPENCODE_IMAGE:?local WildClawBench OpenCode image}"
: "${OPENROUTER_BASE_URL:?judge API base URL, including /v1}"
: "${JUDGE_MODEL:?judge model ID}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-not-required}"

ids=()
while IFS= read -r task || [[ -n "$task" ]]; do
  if [[ -n "$task" ]]; then
    ids+=(--bench-task-id "$task")
  fi
done < "${TASK_IDS_FILE:-task-ids.txt}"

roller collect --bench wildclawbench --bench-root "$WILDCLAWBENCH_ROOT" \
  "${ids[@]}" --bench-runs-per-task "${RUNS_PER_TASK:-3}" \
  --model "$MODEL" --api-base "$MODEL_BASE" --harness opencode \
  --opencode-image "$OPENCODE_IMAGE" --opencode-context-length "$MODEL_CONTEXT_WINDOW" \
  --opencode-max-steps 256 --opencode-output-tokens 32768 --thinking high \
  --task-timeouts --bench-timeout-multiplier 1 --retries 0 --grader-retries 1 \
  --parallel "${PARALLEL:-8}" --grader-parallel "${GRADER_PARALLEL:-1}" \
  --keep-thinking --wire-trace full --capture-calls --capture-debug full \
  --pass-env JUDGE_MODEL --output "${OUTPUT_ROOT:-runs}" \
  --run-name "${RUN_NAME:-wildclawbench-opencode-${MODEL##*/}}" "$@"
