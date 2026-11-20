#!/usr/bin/env bash
# Run one model exposed through Chat Completions.
set -euo pipefail
cd "$(dirname "$0")"
: "${WILDCLAWBENCH_ROOT:?path of the WildClawBench checkout}"
: "${MODEL:?model ID served by the endpoint}"
: "${MODEL_BASE:?Chat Completions API base URL}"
: "${OPENROUTER_BASE_URL:?judge API base URL, including /v1}"
: "${JUDGE_MODEL:?judge model ID}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-not-required}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-not-required}"

execution=()
if [[ "${SANDBOX_PROVIDER:-docker}" == docker ]]; then
  execution+=(--deepseek-image "${DEEPSEEK_IMAGE:?local WildClawBench DeepSeek image}")
else
  execution+=(--sandbox-provider "$SANDBOX_PROVIDER"
    --sandbox-template "${SANDBOX_TEMPLATE:?prebuilt WildClawBench DeepSeek template}")
fi

ids=()
while IFS= read -r task || [[ -n "$task" ]]; do
  if [[ -n "$task" ]]; then
    ids+=(--bench-task-id "$task")
  fi
done < "${TASK_IDS_FILE:-task-ids.txt}"

roller collect --bench wildclawbench --bench-root "$WILDCLAWBENCH_ROOT" \
  "${ids[@]}" --bench-runs-per-task "${RUNS_PER_TASK:-3}" \
  --model "$MODEL" --api-base "$MODEL_BASE" --harness deepseek "${execution[@]}" \
  --task-timeouts --bench-timeout-multiplier 1 --retries 0 --grader-retries 0 \
  --parallel "${PARALLEL:-8}" --grader-parallel "${GRADER_PARALLEL:-2}" \
  --keep-thinking --wire-trace full --capture-calls --pass-env JUDGE_MODEL \
  --output "${OUTPUT_ROOT:-runs}" --run-name "${RUN_NAME:-wildclawbench-deepseek-$MODEL}" \
  "$@"
