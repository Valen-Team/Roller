from roller.harnesses.base import (
    Harness,
    InteractiveHarness,
    InteractiveSession,
    TurnResult,
)
from roller.harnesses.claude_code import ClaudeCodeHarness
from roller.harnesses.claude_code_config import (
    CLI_OPTIONS as CLAUDE_CODE_CLI_OPTIONS,
)
from roller.harnesses.claude_code_config import ClaudeCodeConfig
from roller.harnesses.codex import CodexHarness
from roller.harnesses.codex_config import CLI_OPTIONS as CODEX_CLI_OPTIONS
from roller.harnesses.codex_config import CodexConfig
from roller.harnesses.deepseek import CLI_OPTIONS as DEEPSEEK_CLI_OPTIONS
from roller.harnesses.deepseek import DeepSeekConfig, DeepSeekHarness
from roller.harnesses.hermes import CLI_OPTIONS as HERMES_CLI_OPTIONS
from roller.harnesses.hermes import HermesConfig, HermesHarness
from roller.harnesses.openclaw import OpenClawHarness
from roller.harnesses.openclaw_config import CLI_OPTIONS as OPENCLAW_CLI_OPTIONS
from roller.harnesses.openclaw_config import OpenClawConfig
from roller.harnesses.opencode import CLI_OPTIONS as OPENCODE_CLI_OPTIONS
from roller.harnesses.opencode import OpenCodeConfig, OpenCodeHarness
from roller.harnesses.pi import CLI_OPTIONS as PI_CLI_OPTIONS
from roller.harnesses.pi import PiConfig, PiHarness
from roller.harnesses.stirrup import CLI_OPTIONS as STIRRUP_CLI_OPTIONS
from roller.harnesses.stirrup import StirrupConfig, StirrupHarness

HARNESS_CLI_OPTIONS = {
    "openclaw": OPENCLAW_CLI_OPTIONS,
    "codex": CODEX_CLI_OPTIONS,
    "claude-code": CLAUDE_CODE_CLI_OPTIONS,
    "hermes": HERMES_CLI_OPTIONS,
    "opencode": OPENCODE_CLI_OPTIONS,
    "deepseek": DEEPSEEK_CLI_OPTIONS,
    "pi": PI_CLI_OPTIONS,
    "stirrup": STIRRUP_CLI_OPTIONS,
}

HARNESS_API_PROTOCOL = {
    "codex": "openai-responses",
    "claude-code": "anthropic-messages",
    "hermes": "openai-completions",
    "opencode": "openai-completions",
    "deepseek": "openai-completions",
    "pi": "openai-completions",
    "stirrup": "openai-completions",
}

__all__ = [
    "HARNESS_API_PROTOCOL",
    "HARNESS_CLI_OPTIONS",
    "CodexConfig",
    "CodexHarness",
    "ClaudeCodeConfig",
    "ClaudeCodeHarness",
    "HermesConfig",
    "HermesHarness",
    "OpenCodeConfig",
    "OpenCodeHarness",
    "PiConfig",
    "PiHarness",
    "StirrupConfig",
    "StirrupHarness",
    "DeepSeekConfig",
    "DeepSeekHarness",
    "Harness",
    "InteractiveHarness",
    "InteractiveSession",
    "OpenClawConfig",
    "OpenClawHarness",
    "TurnResult",
]
