#!/usr/bin/env bash
# Run one model exposed through Anthropic Messages.
set -euo pipefail
cd "$(dirname "$0")"
: "${WILDCLAWBENCH_ROOT:?path of the WildClawBench checkout}"
: "${MODEL:?model id served by the endpoint}"
: "${ANTHROPIC_BASE_URL:?Anthropic Messages API base URL}"
: "${OPENROUTER_BASE_URL:?judge API base URL, including /v1}"
: "${JUDGE_MODEL:?judge model id}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-not-required}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-not-required}"

ids=()
while IFS= read -r task; do ids+=(--bench-task-id "$task"); done < "${TASK_IDS_FILE:-task-ids.txt}"
roller collect --bench wildclawbench --bench-root "$WILDCLAWBENCH_ROOT" \
  "${ids[@]}" --bench-runs-per-task "${RUNS_PER_TASK:-3}" \
  --model "$MODEL" --api-base "$ANTHROPIC_BASE_URL" --harness claude-code \
  --claude-code-image "${CLAUDE_CODE_IMAGE:-ghcr.io/valen-team/roller-wildclaw-claude-code:1.3-claude2.1.222}" \
  --task-timeouts --bench-timeout-multiplier 1 --retries 0 --grader-retries 0 \
  --parallel "${PARALLEL:-8}" --grader-parallel "${GRADER_PARALLEL:-2}" \
  --keep-thinking --wire-trace full --capture-calls --pass-env JUDGE_MODEL \
  --pass-env CLAUDE_CODE_MAX_CONTEXT_TOKENS \
  --output "${OUTPUT_ROOT:-runs}" --run-name "${RUN_NAME:-wildclawbench-claude-code-$MODEL}" \
  "$@"
