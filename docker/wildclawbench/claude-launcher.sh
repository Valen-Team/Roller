#!/bin/sh
set -eu

node_dir=/opt/roller-claude-node
install_dir=/tmp/roller-claude-code-${ROLLER_CLAUDE_CODE_VERSION}
cli="$install_dir/node_modules/.bin/claude"
export PATH="$node_dir/bin:$PATH"

if [ ! -x "$cli" ]; then
    node "$node_dir/npm/bin/npm-cli.js" install --prefix "$install_dir" \
        --no-audit --no-fund --no-progress \
        "@anthropic-ai/claude-code@${ROLLER_CLAUDE_CODE_VERSION}" >&2
fi
exec "$cli" "$@"
