from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roller.capture_runtime import CaptureRuntime
from roller.docker_runtime import DockerRuntime
from roller.harnesses.base import InteractiveHarness, InteractiveSession
from roller.harnesses.container import (
    CONTEXT_CONTAINER_PATH,
    INPUTS_CONTAINER_PATH,
    OUTPUTS_CONTAINER_PATH,
    ContainerCommands,
    docker_run_args,
    host_environment,
)
from roller.harnesses.openclaw_config import (
    CACHE_TRACE_CONTAINER_PATH,
    OPENCLAW_STATE_CONTAINER_PATH,
    OPENCLAW_WORKSPACE_LINK,
    TRANSCRIPT_EXPORT_CONTAINER_PATH,
    OpenClawConfig,
    prepare_model_config,
    validate_openclaw_thinking,
)
from roller.harnesses.openclaw_session import OpenClawSession
from roller.models import HarnessResult, RunRequest
from roller.redaction import redact_jsonl_file
from roller.storage import safe_component
from roller.thinking import resolve_thinking
from roller.token_ids import TOKEN_IDS_FILE
from roller.wire_trace import (
    WireTraceProxy,
)

logger = logging.getLogger(__name__)


class OpenClawHarness(ContainerCommands, InteractiveHarness):
    """Run OpenClaw in one disposable Docker container per trajectory."""

    name = "openclaw"

    _MODEL_INJECT_SCRIPT = """
import json
import os
from pathlib import Path
config_path = Path(os.environ['ROLLER_OPENCLAW_CONFIG_PATH'])
models_path = Path('/tmp/roller-models.json')
config = json.loads(config_path.read_text()) if config_path.exists() else {}
config['models'] = json.loads(models_path.read_text())
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(config, indent=2))
""".strip()

    _AUTH_INJECT_SCRIPT = """
import json
import os
from pathlib import Path
path = Path(os.environ['ROLLER_OPENCLAW_AUTH_PATH'])
data = json.loads(path.read_text()) if path.exists() else {'version': 1, 'profiles': {}}
data.setdefault('profiles', {})['openrouter:default'] = {
    'type': 'api_key', 'provider': 'openrouter', 'key': os.environ['ROLLER_OPENROUTER_KEY']
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2))
""".strip()

    _PERSONA_SCRIPT = """
import os
import shutil
from pathlib import Path
workspace = Path(os.environ['ROLLER_WORKSPACE']).resolve()
persona = (workspace / os.environ['ROLLER_PERSONA']).resolve()
persona.relative_to(workspace)
if persona.is_dir():
    for name in ('USER.md', 'MEMORY.md'):
        source = persona / name
        if source.is_file():
            shutil.copyfile(source, Path.home() / name)
""".strip()

    _LATEST_SESSION_SCRIPT = """
import os
from pathlib import Path
roots = [
    Path(os.environ['OPENCLAW_STATE_DIR']) / 'agents/main/sessions',
    Path('/root/.openclaw/agents/main/sessions'),
]
files = sorted(
    {path for root in roots for path in root.glob('*.jsonl')},
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
print(files[0] if files else '')
""".strip()

    _LEGACY_LAST_ASSISTANT_SCRIPT = """
# ROLLER_LEGACY_LAST_ASSISTANT
import json
import os
from pathlib import Path
roots = [
    Path(os.environ['OPENCLAW_STATE_DIR']) / 'agents/main/sessions',
    Path('/root/.openclaw/agents/main/sessions'),
]
preferred = [root / __ROLLER_SESSION_FILE__ for root in roots]
files = [path for path in preferred if path.is_file()]
if not files:
    files = sorted(
        {path for root in roots for path in root.glob('*.jsonl')},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
content = ''
assistant_count = 0
stop_reason = ''
error_message = ''
if files:
    rows = []
    for line in files[0].read_text(errors='replace').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assistant_count = sum(
        isinstance(row, dict)
        and isinstance(row.get('message'), dict)
        and row['message'].get('role') == 'assistant'
        for row in rows
    )
    for row in reversed(rows):
        message = row.get('message') if isinstance(row, dict) else None
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        stop_reason = message.get('stopReason', message.get('stop_reason', ''))
        error_message = message.get('errorMessage', message.get('error_message', ''))
        value = message.get('content')
        if isinstance(value, str):
            content = value
        elif isinstance(value, list):
            content = '\\n'.join(
                block.get('text', '')
                for block in value
                if isinstance(block, dict) and block.get('type') == 'text'
            )
        elif isinstance(value, dict) and isinstance(value.get('text'), str):
            content = value['text']
        break
print(json.dumps({
    'content': content,
    'assistant_count': assistant_count,
    'stop_reason': stop_reason,
    'error': error_message,
}, ensure_ascii=False))
""".strip()

    _SDK_LAST_ASSISTANT_SCRIPT = """
// ROLLER_SDK_LAST_ASSISTANT
const scope = JSON.parse(process.env.ROLLER_SESSION_SCOPE);
const { readLatestAssistantTextByIdentity, readSessionTranscriptEvents } =
  await import('openclaw/plugin-sdk/session-transcript-runtime');
const events = await readSessionTranscriptEvents(scope);
const latest = await readLatestAssistantTextByIdentity(scope);
const assistants = events.filter((event) =>
  event && typeof event === 'object' && event.message &&
  typeof event.message === 'object' && event.message.role === 'assistant'
);
const last = assistants.length ? assistants[assistants.length - 1].message : {};
const blocks = last?.content;
let content = '';
if (typeof blocks === 'string') {
  content = blocks;
} else if (Array.isArray(blocks)) {
  content = blocks
    .filter((block) => block && typeof block === 'object' && block.type === 'text')
    .map((block) => typeof block.text === 'string' ? block.text : '')
    .join('\\n');
} else if (blocks && typeof blocks === 'object' && typeof blocks.text === 'string') {
  content = blocks.text;
}
process.stdout.write(JSON.stringify({
  content,
  assistant_count: assistants.length,
  stop_reason: last?.stopReason ?? last?.stop_reason ?? '',
  error: last?.errorMessage ?? last?.error_message ?? '',
}));
""".strip()

    _SDK_EXPORT_TRANSCRIPT_SCRIPT = """
// ROLLER_SDK_EXPORT_TRANSCRIPT
import fs from 'node:fs';
const scope = JSON.parse(process.env.ROLLER_SESSION_SCOPE);
const { readSessionTranscriptEvents } =
  await import('openclaw/plugin-sdk/session-transcript-runtime');
const events = await readSessionTranscriptEvents(scope);
const payload = events.map((event) => JSON.stringify(event)).join('\\n');
fs.writeFileSync(process.env.ROLLER_TRACE_OUTPUT, payload ? `${payload}\\n` : '');
process.stdout.write(JSON.stringify({ event_count: events.length }));
""".strip()

    _OPENCLAW_NODE_EVAL_SCRIPT = """
if [ -x /opt/roller-node/bin/node ] && [ -f /opt/roller-openclaw/openclaw.mjs ]; then
    cd /opt/roller-openclaw
    exec /opt/roller-node/bin/node --input-type=module -e "$1"
fi
runtime_dir=$(dirname "$(readlink -f "$(command -v openclaw)")")
cd "$runtime_dir"
exec node --input-type=module -e "$1"
""".strip()

    def __init__(
        self,
        config: OpenClawConfig,
        worker_slot: int,
        runtime: DockerRuntime | None = None,
    ) -> None:
        self.config = config
        self.worker_slot = worker_slot
        self.gateway_port = (
            config.gateway_port_start + worker_slot * config.gateway_port_step
        )
        if self.gateway_port > 65535:
            raise ValueError(
                f"worker slot {worker_slot} produces invalid gateway port {self.gateway_port}"
            )
        self.runtime = runtime or DockerRuntime(config.docker_binary)
        self._wire_proxy: WireTraceProxy | None = None
        self.capture = None

    def _start_wire_proxy(
        self, request: RunRequest, output_dir: Path, attempt: int = 1
    ) -> None:
        """Start the wire-trace proxy for this rollout.

        The proxy runs when wire tracing is active, when thinking must be
        stripped from request history, when Chat Completions reasoning fields
        need normalization.
        """
        self.capture = (
            CaptureRuntime(output_dir, attempt, "openclaw", self.config.capture_debug)
            if self.config.capture_calls
            else None
        )
        self._wire_proxy = None
        needs_proxy = (
            self.config.wire_trace_mode != "off"
            or not self.config.keep_thinking
            or self.config.model_config is not None
        )
        if not needs_proxy:
            return
        proxy_required = (
            self.config.wire_trace_mode == "full"
            or self.config.capture_calls
            or not self.config.keep_thinking
        )
        if self.config.model_config is None:
            if proxy_required:
                raise RuntimeError(
                    "wire tracing/stripping requires --model-config with the selected "
                    "provider's baseUrl"
                )
            logger.debug(
                "[%s] tools schema capture skipped: no model config was provided",
                request.task.id,
            )
            return
        providers = self.config.model_config.get("providers")
        provider_name = request.model.split("/", 1)[0]
        provider = (
            providers.get(provider_name) if isinstance(providers, Mapping) else None
        )
        upstream = provider.get("baseUrl") if isinstance(provider, Mapping) else None
        if not isinstance(upstream, str) or not upstream.strip():
            if proxy_required:
                raise RuntimeError(
                    f"wire tracing/stripping requires baseUrl for provider {provider_name!r}"
                )
            logger.debug(
                "[%s] tools schema capture skipped: provider %r has no baseUrl",
                request.task.id,
                provider_name,
            )
            return
        if self.config.wire_trace_mode == "full" or self.config.capture_calls:
            log_path = output_dir / "wire_trace.jsonl"
            log_mode = "full"
        elif self.config.wire_trace_mode == "actf":
            log_path = output_dir / "wire_trace.jsonl"
            log_mode = "tools"
        else:  # Internal-only native thinking-history compatibility proxy.
            log_path = output_dir / "rollout" / "wire_strip.jsonl"
            log_mode = "tools"
        token_ids_path = output_dir / TOKEN_IDS_FILE
        token_ids_path.unlink(missing_ok=True)
        try:
            self._wire_proxy = WireTraceProxy(
                upstream,
                log_path,
                mode=log_mode,
                capture_recorder=self.capture.recorder if self.capture else None,
                capture_identity_header="x-roller-native-session-id"
                if self.capture
                else None,
                capture_run_header="x-roller-native-run-id" if self.capture else None,
                strip_thinking=not self.config.keep_thinking,
                normalize_reasoning_replay=self.config.keep_thinking,
                token_ids_path=token_ids_path if self.config.token_ids else None,
            )
            self._wire_proxy.start()
        except Exception as exc:
            if self._wire_proxy is not None:
                self._stop_wire_proxy()
            if proxy_required:
                raise RuntimeError(f"wire trace proxy failed to start: {exc}") from exc
            logger.warning(
                "[%s] tools schema capture skipped: %s", request.task.id, exc
            )
            self._wire_proxy = None

    def _stop_wire_proxy(self) -> None:
        if self._wire_proxy is not None:
            try:
                self._wire_proxy.stop()
            except Exception as exc:
                logger.warning("wire trace proxy shutdown failed: %s", exc)
            self._wire_proxy = None

    def _environment(self, request: RunRequest) -> dict[str, str]:
        environment = host_environment(request.task, self.config.pass_env)

        # Compatibility with Interns2_data's host-side variable name.
        if "OPENROUTER_BASE_URL" not in environment and os.environ.get(
            "OPENROUTER_BASE_URL_OPENCLAW"
        ):
            environment["OPENROUTER_BASE_URL"] = os.environ[
                "OPENROUTER_BASE_URL_OPENCLAW"
            ]
        return environment

    def _start_container(
        self, container: str, request: RunRequest, image: str | None = None
    ) -> dict[str, str]:
        environment = self._environment(request)
        # Keep Roller independent of the image's default user/HOME. Official
        # OpenClaw images run as ``node`` while legacy benchmark images run as
        # ``root``; both can write this per-container state directory.
        args = docker_run_args(
            self.name,
            container,
            self.config,
            environment,
            request,
            extra_env={"OPENCLAW_STATE_DIR": OPENCLAW_STATE_CONTAINER_PATH},
        )
        args.extend(
            [image or self.config.docker_image, "/bin/bash", "-c", "tail -f /dev/null"]
        )
        self._checked(args, "container startup", timeout=60)
        return environment

    def _exec_checked(
        self,
        container: str,
        command: Sequence[str],
        description: str,
        *,
        env: Mapping[str, str] | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """``_checked`` for one command in the container."""

        return self._checked(
            self.runtime.exec_args(container, command, env=env, workdir=workdir),
            description,
            timeout,
        )

    def _prepare_workspace(self, container: str, request: RunRequest) -> None:
        workspace = self.config.workspace_container_path
        output_root = f"{workspace}/{request.task.output_dir}"
        if request.bundle is not None:
            if request.bundle.legacy_workspace or request.task.input_mode == "copy":
                setup = (
                    f"mkdir -p {workspace} && cp -a {INPUTS_CONTAINER_PATH}/. {workspace}/ && "
                    f"chmod -R u+w {workspace} && rm -rf {workspace}/context && "
                    f"ln -s {CONTEXT_CONTAINER_PATH} {workspace}/context"
                )
            else:
                setup = f"mkdir -p {workspace}"
        elif request.task.workspace is not None:
            setup = (
                f"mkdir -p {workspace} && cp -a /roller-input/. {workspace}/ && "
                f"chmod -R u+w {workspace}"
            )
        else:
            setup = f"mkdir -p {workspace}"
        quoted_output_root = shlex.quote(output_root)
        script = (
            f"{setup} && rm -rf {quoted_output_root} && mkdir -p {quoted_output_root} && "
            f"mkdir -p /roller/task && rm -rf {OUTPUTS_CONTAINER_PATH} && "
            f"ln -s {quoted_output_root} {OUTPUTS_CONTAINER_PATH}"
        )
        self._exec_checked(container, ["/bin/bash", "-c", script], "workspace setup")
        link_script = (
            f"mkdir -p {OPENCLAW_STATE_CONTAINER_PATH} /root/.openclaw && "
            f"rm -rf {OPENCLAW_STATE_CONTAINER_PATH}/workspace && "
            f"ln -s {workspace} {OPENCLAW_STATE_CONTAINER_PATH}/workspace && "
            f"rm -rf {OPENCLAW_WORKSPACE_LINK} && "
            f"ln -s {workspace} {OPENCLAW_WORKSPACE_LINK}"
        )
        self._exec_checked(
            container,
            ["/bin/bash", "-c", link_script],
            "OpenClaw workspace link",
        )

    @staticmethod
    def _state_dir_for_version(version: str) -> str:
        # OpenClaw 2026.3.11 predates OPENCLAW_STATE_DIR. Keep its baked image
        # and JSONL compatibility path while newer runtimes use isolated state.
        if re.search(r"(?:^|\s)2026\.3\.11(?:\s|$)", version):
            return "/root/.openclaw"
        return OPENCLAW_STATE_CONTAINER_PATH

    def _effective_thinking(self, request: RunRequest) -> str | None:
        return resolve_thinking(request.task.thinking, self.config.thinking)

    def _inject_models(
        self,
        container: str,
        request: RunRequest,
        state_dir: str,
    ) -> None:
        if self.config.model_config is None:
            return
        thinking = self._effective_thinking(request)
        config_to_write = prepare_model_config(
            self.config.model_config,
            request.model,
            proxy=self._wire_proxy,
            enable_reasoning=thinking not in (None, "off"),
            request_timeout_seconds=self.config.model_request_timeout_seconds,
            max_tokens=self.config.model_max_tokens,
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".json", delete=False
            ) as stream:
                json.dump(config_to_write, stream, ensure_ascii=False)
                temporary_path = Path(stream.name)
            self._checked(
                self.runtime.copy_in_args(
                    temporary_path, container, "/tmp/roller-models.json"
                ),
                "model config copy",
            )
            self._exec_checked(
                container,
                ["python3", "-c", self._MODEL_INJECT_SCRIPT],
                "model config injection",
                env={"ROLLER_OPENCLAW_CONFIG_PATH": f"{state_dir}/openclaw.json"},
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _configure(
        self, container: str, request: RunRequest, environment: Mapping[str, str]
    ) -> str:
        version = self._exec_checked(
            container, ["openclaw", "--version"], "OpenClaw version"
        ).stdout.strip()
        state_dir = self._state_dir_for_version(version)
        thinking = self._effective_thinking(request)
        validate_openclaw_thinking(version, thinking)
        self._inject_models(container, request, state_dir)
        self._exec_checked(
            container,
            ["openclaw", "models", "set", request.model],
            "model setup",
        )
        self._exec_checked(
            container,
            [
                "openclaw",
                "config",
                "set",
                "gateway.port",
                str(self.gateway_port),
            ],
            "gateway port setup",
        )
        self._exec_checked(
            container,
            ["openclaw", "config", "set", "gateway.mode", "local"],
            "gateway mode setup",
        )
        self._exec_checked(
            container,
            [
                "openclaw",
                "config",
                "set",
                "gateway.bind",
                "loopback",
            ],
            "gateway bind setup",
        )
        self._exec_checked(
            container,
            [
                "openclaw",
                "config",
                "set",
                "agents.defaults.workspace",
                OPENCLAW_WORKSPACE_LINK,
            ],
            "OpenClaw workspace setup",
        )
        if thinking:
            self._exec_checked(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "agents.defaults.thinkingDefault",
                    thinking,
                ],
                "thinking mode setup",
            )
        if self.config.search_provider:
            self._exec_checked(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "tools.web.search.provider",
                    self.config.search_provider,
                ],
                "search provider setup",
            )
        if self.config.capture_cache_trace:
            # Persist the fully assembled system prompt (per LLM call) via
            # OpenClaw's diagnostics.cacheTrace so downstream consumers never
            # need to reconstruct or hardcode it.
            self._exec_checked(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "diagnostics.cacheTrace.enabled",
                    "true",
                ],
                "cache trace setup",
            )
            self._exec_checked(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "diagnostics.cacheTrace.filePath",
                    CACHE_TRACE_CONTAINER_PATH,
                ],
                "cache trace path setup",
            )
        # OpenClaw masks credential-shaped text (``RV_9fc…1812``) when it
        # persists a session, and later turns rebuild context from that copy.
        # Task text must survive verbatim; trajectories are sensitive anyway.
        try:
            self._exec_checked(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "logging.redactSensitive",
                    "off",
                ],
                "transcript redaction setup",
            )
        except RuntimeError as exc:
            # Releases that retired the key reject it; keep running.
            logger.warning(
                "[%s] OpenClaw %s kept logging.redactSensitive; its session transcript "
                "may abbreviate credential-shaped task text: %s",
                request.task.id,
                version,
                exc,
            )
        # OpenClaw polls its main session on a timer and when a sub-agent
        # finishes. Each poll is a model call outside the task, and its
        # session shows up as a second root participant in the capture.
        try:
            self._exec_checked(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "agents.defaults.heartbeat.every",
                    "0m",
                ],
                "heartbeat setup",
            )
        except RuntimeError as exc:
            logger.warning(
                "[%s] OpenClaw %s kept its heartbeat polls; they add model calls "
                "outside the task: %s",
                request.task.id,
                version,
                exc,
            )
        if self.config.set_image_model:
            image_result = self.runtime.exec(
                container,
                [
                    "openclaw",
                    "config",
                    "set",
                    "agents.defaults.imageModel.primary",
                    request.model,
                ],
                timeout=self._bounded(None, "image model setup"),
            )
            if image_result.returncode != 0:
                logger.warning(
                    "[%s] image model setup was not supported", request.task.id
                )
        if environment.get("OPENROUTER_API_KEY"):
            self._exec_checked(
                container,
                ["python3", "-c", self._AUTH_INJECT_SCRIPT],
                "OpenRouter auth setup",
                env={
                    "ROLLER_OPENROUTER_KEY": environment["OPENROUTER_API_KEY"],
                    "ROLLER_OPENCLAW_AUTH_PATH": (
                        f"{state_dir}/agents/main/agent/auth-profiles.json"
                    ),
                },
            )
        for key, value in self.config.config_set:
            # ``--`` keeps a value such as ``-1`` from being read as an option.
            self._exec_checked(
                container,
                ["openclaw", "config", "set", "--", key, value],
                f"OpenClaw config setting {key!r}",
            )
        return version

    def _prepare_context_and_setup(self, container: str, request: RunRequest) -> None:
        if request.bundle is not None:
            self._exec_checked(
                container,
                [
                    "/bin/bash",
                    "-c",
                    f'cp -a {CONTEXT_CONTAINER_PATH}/. "$HOME"/ && '
                    f"cp -a {CONTEXT_CONTAINER_PATH}/. "
                    f"{self.config.workspace_container_path}/",
                ],
                "context injection",
            )
        else:
            persona = request.task.raw.get("persona")
            if isinstance(persona, str) and persona.strip():
                self._exec_checked(
                    container,
                    [
                        "python3",
                        "-c",
                        self._PERSONA_SCRIPT,
                    ],
                    "persona setup",
                    env={
                        "ROLLER_WORKSPACE": self.config.workspace_container_path,
                        "ROLLER_PERSONA": persona.strip(),
                    },
                )
        for command in request.task.setup_commands:
            self._exec_checked(
                container,
                [
                    "/bin/bash",
                    "-lc",
                    command,
                ],
                f"task setup command {command!r}",
                env={"ROLLER_GATEWAY_PORT": str(self.gateway_port)},
                workdir=self.config.workspace_container_path,
            )

    def _gateway_port_probe(self) -> str:
        """Python snippet exiting 0 when something accepts on the gateway port."""
        return (
            "import socket,sys; s=socket.socket(); s.settimeout(.2); "
            f"sys.exit(s.connect_ex(('127.0.0.1',{self.gateway_port})))"
        )

    def _ensure_gateway_port_free(self, container: str) -> None:
        """Refuse to start a gateway on a port something already listens on.

        Under host networking the readiness probe cannot tell this rollout's
        gateway from another one on the same port: the agent would connect to
        that gateway, be refused, and quietly run without its own.
        """
        if self.config.network != "host":
            return  # the gateway does not share the host's ports
        try:
            result = self.runtime.exec(
                container,
                ["python3", "-c", self._gateway_port_probe()],
                timeout=self._bounded(10, "OpenClaw gateway port check"),
            )
        except subprocess.TimeoutExpired:
            return  # inconclusive; the readiness wait still bounds startup
        if result.returncode == 0:
            raise RuntimeError(
                f"OpenClaw gateway port {self.gateway_port} is already in use on this "
                "host by another server, a run of an older Roller version, or a "
                "container left over from an earlier rollout of this run (check "
                "`docker ps`)"
            )

    def _wait_for_gateway(self, container: str, process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + self._bounded(
            self.config.gateway_startup_timeout, "OpenClaw gateway startup"
        )
        probe = self._gateway_port_probe()
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"OpenClaw gateway exited early with code {process.returncode}"
                )
            try:
                result = self.runtime.exec(
                    container, ["python3", "-c", probe], timeout=2
                )
            except subprocess.TimeoutExpired:
                # Docker exec can be slow while other Harness containers start.
                # A single probe timeout must not discard the startup budget.
                continue
            if result.returncode == 0:
                return
            time.sleep(0.25)
        self._bounded(None, "OpenClaw gateway startup")  # the budget was the limit
        # RuntimeError, not TimeoutError: callers report TimeoutError as the
        # task budget running out, which this startup failure is not.
        raise RuntimeError(
            f"OpenClaw gateway did not listen on port {self.gateway_port} within "
            f"{self.config.gateway_startup_timeout:g}s"
        )

    def _prompts(self, request: RunRequest) -> tuple[str, ...]:
        return request.task.prompts

    def _session_scope(self, session_id: str | None = None) -> dict[str, str]:
        session_id = session_id or self.config.session_id
        return {
            "agentId": "main",
            "sessionId": session_id,
            "sessionKey": f"agent:main:explicit:{session_id}",
        }

    def _run_transcript_sdk(
        self,
        container: str,
        script: str,
        *,
        output_path: str | None = None,
        state_dir: str | None = None,
        session_id: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            "ROLLER_SESSION_SCOPE": json.dumps(
                self._session_scope(session_id), separators=(",", ":")
            )
        }
        if output_path is not None:
            env["ROLLER_TRACE_OUTPUT"] = output_path
        if state_dir is not None:
            # OpenClaw 2026.3.11 ignores OPENCLAW_STATE_DIR for gateway
            # commands but the transcript SDK follows it. Keep both sides on
            # the same state directory when using the legacy image.
            env["OPENCLAW_STATE_DIR"] = state_dir
        return self.runtime.exec(
            container,
            [
                "/bin/sh",
                "-lc",
                self._OPENCLAW_NODE_EVAL_SCRIPT,
                "roller-transcript-sdk",
                script,
            ],
            env=env,
        )

    def _latest_assistant_content(
        self,
        container: str,
        *,
        state_dir: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, int]:
        result = self._run_transcript_sdk(
            container,
            self._SDK_LAST_ASSISTANT_SCRIPT,
            state_dir=state_dir,
            session_id=session_id,
        )
        if result.returncode != 0:
            logger.debug(
                "OpenClaw transcript SDK unavailable; using legacy JSONL response reader: %s",
                self._error_text(result),
            )
            script = self._LEGACY_LAST_ASSISTANT_SCRIPT.replace(
                "__ROLLER_SESSION_FILE__",
                repr(f"{session_id or self.config.session_id}.jsonl"),
            )
            result = self._exec_checked(
                container,
                ["python3", "-c", script],
                "assistant response extraction",
            )
        try:
            payload = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "assistant response extraction returned invalid JSON"
            ) from exc
        content = payload.get("content") if isinstance(payload, dict) else None
        assistant_count = (
            payload.get("assistant_count") if isinstance(payload, dict) else None
        )
        stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
        if stop_reason in {"aborted", "error"}:
            detail = payload.get("error") if isinstance(payload, dict) else None
            raise RuntimeError(
                detail
                if isinstance(detail, str) and detail
                else f"OpenClaw {stop_reason}"
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenClaw turn produced no visible assistant response")
        if isinstance(assistant_count, bool) or not isinstance(assistant_count, int):
            raise RuntimeError("assistant response extraction returned invalid count")
        return content.strip(), assistant_count

    def _collect_trace(
        self,
        container: str,
        output_dir: Path,
        attempt: int,
        *,
        state_dir: str | None = None,
        session_ids: Sequence[str] | None = None,
    ) -> Path | None:
        rollout_dir = output_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        attempt_trace_path = rollout_dir / f"trace.attempt-{attempt}.jsonl"
        trace_path = rollout_dir / "trace.jsonl"
        attempt_trace_path.unlink(missing_ok=True)

        container_trace = self._export_combined_transcript(
            container,
            session_ids or (self.config.session_id,),
            state_dir=state_dir,
        )
        result = (
            self.runtime.copy_out(container, container_trace, attempt_trace_path)
            if container_trace
            else subprocess.CompletedProcess([], 1, "", "")
        )
        if result.returncode != 0:
            latest = self.runtime.exec(
                container,
                ["python3", "-c", self._LATEST_SESSION_SCRIPT],
            )
            fallback_path = latest.stdout.strip() if latest.returncode == 0 else ""
            if fallback_path:
                result = self.runtime.copy_out(
                    container, fallback_path, attempt_trace_path
                )
        if (
            result.returncode != 0
            or not attempt_trace_path.is_file()
            or attempt_trace_path.stat().st_size == 0
        ):
            return None
        shutil.copyfile(attempt_trace_path, trace_path)
        return trace_path

    def _export_combined_transcript(
        self,
        container: str,
        session_ids: Sequence[str],
        *,
        state_dir: str | None = None,
    ) -> str | None:
        """Export one or more OpenClaw sessions into one grader-visible JSONL."""

        sources: list[str] = []
        session_root = state_dir or OPENCLAW_STATE_CONTAINER_PATH
        for index, session_id in enumerate(session_ids):
            destination = f"{TRANSCRIPT_EXPORT_CONTAINER_PATH}.{index}"
            result = self._run_transcript_sdk(
                container,
                self._SDK_EXPORT_TRANSCRIPT_SCRIPT,
                output_path=destination,
                state_dir=state_dir,
                session_id=session_id,
            )
            if result.returncode == 0:
                sources.append(destination)
                continue
            logger.debug(
                "OpenClaw transcript SDK unavailable for %s; using legacy JSONL: %s",
                session_id,
                self._error_text(result),
            )
            sources.append(f"{session_root}/agents/main/sessions/{session_id}.jsonl")
        if not sources:
            return None
        quoted = " ".join(shlex.quote(path) for path in sources)
        combined = shlex.quote(TRANSCRIPT_EXPORT_CONTAINER_PATH)
        result = self.runtime.exec(
            container,
            [
                "/bin/sh",
                "-c",
                f"cat {quoted} > {combined}",
            ],
            timeout=60,
        )
        return TRANSCRIPT_EXPORT_CONTAINER_PATH if result.returncode == 0 else None

    def _collect_cache_trace(self, container: str, output_dir: Path) -> Path | None:
        """Copy the OpenClaw cache-trace file (system prompts per LLM call) into the sample."""
        destination = output_dir / "cache_trace.jsonl"
        if not self.config.capture_cache_trace:
            return None
        destination.unlink(missing_ok=True)
        result = self.runtime.copy_out(
            container, CACHE_TRACE_CONTAINER_PATH, destination
        )
        if result.returncode != 0 or not destination.is_file():
            return None
        try:
            redact_jsonl_file(destination)
        except Exception:
            # Full mode retains this file. Never leave an unsanitized credential-
            # bearing cache trace behind when the redaction pass fails.
            destination.unlink(missing_ok=True)
            raise
        return destination

    def _collect_legacy_artifacts(
        self, container: str, output_dir: Path
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        artifact_root = output_dir / "results" / "legacy"
        for index, container_path in enumerate(self.config.collect_paths):
            destination = (
                artifact_root
                / f"{index:02d}_{safe_component(Path(container_path).name or 'root')}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = self.runtime.copy_out(container, container_path, destination)
            records.append(
                {
                    "container_path": container_path,
                    "output_path": str(destination) if result.returncode == 0 else None,
                    "error": None
                    if result.returncode == 0
                    else self._error_text(result),
                }
            )
        return records

    @staticmethod
    def _stop_process(process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def create_session(
        self,
        request: RunRequest,
        output_dir: Path,
        attempt: int,
        *,
        collect_outputs: bool = True,
    ) -> InteractiveSession:
        return OpenClawSession(
            self,
            request,
            output_dir,
            attempt,
            collect_outputs=collect_outputs,
        )

    def run(self, request: RunRequest, output_dir: Path, attempt: int) -> HarnessResult:
        session = self.create_session(request, output_dir, attempt)
        status = "error"
        error: str | None = None
        started = time.monotonic()
        try:
            session.start()
            for index, prompt in enumerate(self._prompts(request)):
                remaining = request.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"task exceeded {request.timeout_seconds}s")
                if (
                    index < len(request.task.new_session)
                    and request.task.new_session[index]
                ):
                    session.reset()
                session.send(prompt, remaining)
            status = "ok"
        except (TimeoutError, subprocess.TimeoutExpired):
            status = "timeout"
            error = f"task exceeded {request.timeout_seconds}s"
        except Exception as exc:
            status = "error"
            error = str(exc)
        return session.finish(status, error)
