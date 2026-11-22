#!/usr/bin/env bash
# ZeroBench with Claude Code in a remote sandbox created without egress.
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL:?model ID served by the endpoint}"
: "${MODEL_BASE:?Chat Completions API base URL}"
: "${SANDBOX_PROVIDER:?name a provider registered under roller.sandbox_providers}"
: "${SANDBOX_TEMPLATE:?prebuilt environment that mounts the Claude Code release}"
# The CLI gets the provider's relay token; this key stays on the collecting host.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-not-required}"

data="${ZEROBENCH_DATA:-zerobench-data}"
mkdir -p "$data"

roller collect --bench zerobench --bench-root "$data" \
  --bench-variant "${ZEROBENCH_VARIANT:-zerobench}" \
  --samples "${SAMPLES:-5}" \
  --model "$MODEL" --harness claude-code \
  --claude-api-base "$MODEL_BASE" --claude-chat-completions \
  --claude-tool-images-inline \
  --sandbox-provider "$SANDBOX_PROVIDER" --sandbox-template "$SANDBOX_TEMPLATE" \
  --thinking max --claude-max-turns 100 \
  --retries 0 --parallel "${PARALLEL:-4}" \
  --keep-thinking --wire-trace full --capture-calls --token-ids \
  --output "${OUTPUT_ROOT:-runs}" \
  --run-name "${RUN_NAME:-zerobench-claude-code-$MODEL}" \
  "$@"
