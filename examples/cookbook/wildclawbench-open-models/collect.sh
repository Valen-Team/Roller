#!/usr/bin/env bash
# Collect and grade the WildClawBench tasks of task-ids.txt on four self-hosted
# models in one Roller run. Arguments are passed on to `roller collect`, for
# example `./collect.sh --dry-run` or `./collect.sh --resume`.
set -euo pipefail
cd "$(dirname "$0")"

: "${WILDCLAWBENCH_ROOT:?path of the WildClawBench checkout}"
: "${ROLLER_INFER_BASE_URL:?}" "${ROLLER_INFER_API_KEY:?}"

# A separate judge endpoint can be supplied when collecting several models.
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-$ROLLER_INFER_BASE_URL}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$ROLLER_INFER_API_KEY}"
export JUDGE_MODEL="${JUDGE_MODEL:-glm-5.3-flash}"

# shellcheck disable=SC2046
roller collect --bench wildclawbench --bench-root "$WILDCLAWBENCH_ROOT" \
  $(sed 's/^/--bench-task-id /' task-ids.txt) \
  --bench-runs-per-task 3 \
  --model vllm/glm-5.3-flash \
  --model vllm/deepseek-v4.1-flash-1017 \
  --model vllm/kimi-k3 --model vllm/kimi-k3.1-pro \
  --model-config models.json \
  --harness openclaw --openclaw-image "${OPENCLAW_IMAGE:-ghcr.io/valen-team/roller-wildclaw-base:1.3-clean.1}" \
  --openclaw-config-set tools.web.search.enabled=false \
  --openclaw-max-tokens 32768 --task-timeouts --retries 0 --grader-retries 0 \
  --parallel "${PARALLEL:-8}" --grader-parallel "${GRADER_PARALLEL:-2}" \
  --keep-thinking --wire-trace full \
  --output "${OUTPUT_ROOT:-runs}" \
  --run-name wildclawbench-open-models \
  "$@"
