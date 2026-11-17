#!/usr/bin/env bash
# Run one Responses-compatible model; launch separate invocations for more models.
set -euo pipefail
cd "$(dirname "$0")"
: "${WILDCLAWBENCH_ROOT:?path of the WildClawBench checkout}"
: "${MODEL:?model id served by the endpoint}"
: "${OPENAI_BASE_URL:?Responses API base URL, including /v1}"
: "${OPENROUTER_BASE_URL:?judge API base URL, including /v1}"
: "${JUDGE_MODEL:?judge model id}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-not-required}"

compat=()
if [[ "${TOOL_IMAGES_AS_USER:-0}" == 1 ]]; then
  compat+=(--codex-tool-images-as-user)
fi
if [[ -n "${MODEL_CONTEXT_WINDOW:-}" ]]; then
  compat+=(--codex-config-set "model_context_window=$MODEL_CONTEXT_WINDOW")
fi
ids=()
while IFS= read -r task; do ids+=(--bench-task-id "$task"); done < "${TASK_IDS_FILE:-task-ids.txt}"
roller collect --bench wildclawbench --bench-root "$WILDCLAWBENCH_ROOT" \
  "${ids[@]}" --bench-runs-per-task "${RUNS_PER_TASK:-3}" \
  --model "$MODEL" --api-base "$OPENAI_BASE_URL" \
  --harness codex \
  --codex-image "${CODEX_IMAGE:-ghcr.io/valen-team/roller-wildclaw-codex:1.3-codex0.149.0}" \
  --no-codex-websockets --codex-web-search disabled --codex-image-detail auto \
  --codex-config-set features.enable_request_compression=false \
  --task-timeouts --retries 0 --grader-retries 0 \
  --parallel "${PARALLEL:-8}" --grader-parallel "${GRADER_PARALLEL:-2}" \
  --keep-thinking --wire-trace full --pass-env JUDGE_MODEL \
  --output "${OUTPUT_ROOT:-runs}" --run-name "${RUN_NAME:-wildclawbench-codex}" \
  "${compat[@]}" "$@"
