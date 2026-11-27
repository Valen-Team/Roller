"""Wire-level request recorder and thinking-history compatibility proxy.

Roller points the harness's model ``baseUrl`` at a local instance of this proxy.
The proxy records what the real server would receive — full tools schema,
sampling parameters and (optionally) complete message bodies — then forwards
the request. Depending on configuration, it can also strip historical thinking
from requests or normalize equivalent Chat Completions reasoning response fields
before OpenClaw receives them. Because interception happens at the
OpenAI-compatible HTTP layer, it works for harness versions that honour
``baseUrl``.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib import request as urllib_request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from roller.anthropic_pdf import render_anthropic_pdfs
from roller.blobs import BLOBS_DIR, store_blobs
from roller.capture import CallRecorder
from roller.codex_tools import tool_snapshot
from roller.response_metadata import ResponseMetadata
from roller.token_ids import TokenIds, TokenLog
from roller.tool_id_identity import ToolIdRegistry
from roller.wire_transforms import (
    BODY_KEYS,
    ENCRYPTED_CONTENT_ERROR,
    ENCRYPTED_CONTENT_RETRY_DELAY_SECONDS,
    HOP_BY_HOP_HEADERS,
    RESPONSES_IMAGE_DETAILS,
    _image_digest,
    _is_encrypted_content_rejection,
    _request_image_inputs,
    _rewrite_event,
    _rewrite_json,
    _sse,
    fill_responses_image_detail,
    move_responses_tool_images,
    normalize_chat_completion_reasoning_response,
    strip_thinking_from_body,
    wire_text_digest,
)


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def _record(self, body: bytes | None) -> int:
        proxy: WireTraceProxy = self.server.proxy  # type: ignore[attr-defined]
        capture_responses_tools = (
            proxy.responses_tools_path is not None
            and self.command == "POST"
            and urlsplit(self.path).path.rstrip("/").endswith("/responses")
        )
        entry: dict[str, Any] = {
            "seq": proxy.next_seq(responses_request=capture_responses_tools),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "kind": "request",
            "method": self.command,
            "path": self.path,
        }
        upstream_path = proxy.upstream_path(self.command, self.path)
        if upstream_path != self.path:
            entry["upstream_path"] = upstream_path
        dropped_headers = sorted(
            name for name in self.headers if name.lower() in proxy.drop_request_headers
        )
        if dropped_headers:
            entry["dropped_request_headers"] = dropped_headers
        parsed: dict[str, Any] | None = None
        if body:
            try:
                candidate = json.loads(body)
                parsed = candidate if isinstance(candidate, dict) else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
        if capture_responses_tools and parsed is None:
            raise ValueError("Cannot capture tools from a non-object Responses request")
        if parsed is not None:
            entry["model"] = parsed.get("model")
            if capture_responses_tools:
                proxy.write_tools(
                    {
                        "kind": "codex_tools",
                        "seq": entry["seq"],
                        "snapshot": tool_snapshot(parsed),
                    }
                )
            if proxy.mode == "full":
                entry["body"] = parsed = store_blobs(parsed, proxy.blob_dir)
            else:
                params = {k: v for k, v in parsed.items() if k not in BODY_KEYS}
                entry["params"] = params
                image_inputs, image_texts = _request_image_inputs(parsed)
                if image_inputs:
                    # Later requests resend the whole history; keep each
                    # image's bytes once and reference repeats by digest.
                    entry["image_inputs"] = proxy.image_refs(image_inputs)
                    entry["image_text_digests"] = [
                        wire_text_digest(text) for text in image_texts
                    ]
                    raw_tools = parsed.get("tools")
                    entry["n_tools"] = (
                        len(raw_tools) if isinstance(raw_tools, list) else 0
                    )
                messages = parsed.get("messages")
                if isinstance(messages, list):
                    entry["n_messages"] = len(messages)
                tools = parsed.get("tools")
                if isinstance(tools, list) and tools:
                    fingerprint = json.dumps(
                        tools,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if fingerprint not in proxy.tool_schema_fingerprints:
                        proxy.tool_schema_fingerprints.add(fingerprint)
                        entry["kind"] = "tools"
                        entry["tools"] = tools
                        entry["n_tools"] = len(tools)
        proxy.write(entry)
        if (
            proxy.capture_recorder is not None
            and parsed is not None
            and self.command == "POST"
            and urlsplit(self.path)
            .path.rstrip("/")
            .endswith(("/messages", "/chat/completions", "/responses"))
        ):
            native_session = (
                self.headers.get(proxy.capture_identity_header)
                if proxy.capture_identity_header
                else None
            )
            native_run = (
                self.headers.get(proxy.capture_run_header)
                if proxy.capture_run_header
                else None
            )
            proxy.capture_recorder.begin(
                entry["seq"],
                proxy.forward_path(self.path),
                parsed,
                native_identity={
                    **({"session_id": native_session} if native_session else {}),
                    **({"run_id": native_run} if native_run else {}),
                }
                or None,
            )
        return entry["seq"]

    def _handle(self) -> None:
        proxy: WireTraceProxy = self.server.proxy  # type: ignore[attr-defined]
        length = self.headers.get("Content-Length")
        body = self.rfile.read(int(length)) if length and length.isdigit() else None
        if body and proxy.strip_thinking:
            try:
                parsed = strip_thinking_from_body(json.loads(body))
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if (
            body
            and (proxy.responses_image_detail or proxy.responses_tool_images_as_user)
            and self.command == "POST"
            and urlsplit(self.path).path.rstrip("/").endswith("/responses")
        ):
            try:
                parsed = json.loads(body)
                # Requests without such an image keep their original bytes.
                changed = False
                if proxy.responses_image_detail:
                    changed = fill_responses_image_detail(
                        parsed, proxy.responses_image_detail
                    )
                if proxy.responses_tool_images_as_user:
                    changed = move_responses_tool_images(parsed) or changed
                if changed:
                    body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if (
            body
            and proxy.anthropic_pdf_as_images
            and self.command == "POST"
            and urlsplit(self.path)
            .path.rstrip("/")
            .endswith(("/messages", "/messages/count_tokens"))
        ):
            try:
                parsed = json.loads(body)
                if render_anthropic_pdfs(parsed):
                    body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        request_seq = None
        try:
            request_seq = self._record(body)
        except Exception as exc:
            if proxy.responses_tools_path is not None:
                proxy.recording_error = str(exc)
            if proxy.capture_recorder is not None:
                proxy.capture_recorder.error = str(exc)
        self.close_connection = True
        upstream_path = proxy.upstream_path(self.command, self.path)
        bridged = upstream_path != proxy.forward_path(self.path)
        # The Chat Completions request the proxy writes itself, if it does.
        bridge = proxy.chat_bridge
        request = json.loads(body) if bridged else None
        chat = (
            bridge.chat_request(request, **proxy.chat_bridge_options)
            if bridged
            else None
        )
        tokens = None
        if (
            proxy.token_log is not None
            and self.command == "POST"
            and urlsplit(upstream_path).path.rstrip("/").endswith("/chat/completions")
        ):
            chat = chat or json.loads(body)
            tokens = TokenIds(chat)
        if chat is not None:
            body = json.dumps(chat, ensure_ascii=False).encode("utf-8")
        url = proxy.upstream_origin + upstream_path
        outgoing = urllib_request.Request(url, data=body, method=self.command)
        for key, value in self.headers.items():
            normalized_key = key.lower()
            if (
                normalized_key not in HOP_BY_HOP_HEADERS
                and normalized_key != "host"
                and normalized_key not in proxy.drop_request_headers
            ):
                outgoing.add_header(key, value)
        if proxy.tool_id_mappings_path is not None or chat is not None:
            # The proxy reads the JSON of these responses, so uncompressed.
            outgoing.add_header("Accept-Encoding", "identity")
        if bridged and self.headers.get("x-api-key"):
            outgoing.add_header("Authorization", f"Bearer {self.headers['x-api-key']}")
        try:
            response = proxy.open_upstream(outgoing, request_seq)
        except Exception as exc:
            if (
                proxy.capture_recorder is not None
                and request_seq in proxy.capture_recorder.request_seqs
            ):
                proxy.capture_recorder.finish(
                    request_seq,
                    status=None,
                    content_type="",
                    content_encoding="",
                    complete=False,
                    error=f"upstream transport error: {type(exc).__name__}",
                )
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        status = getattr(response, "status", None)
        if not isinstance(status, int):
            status = getattr(response, "code", None)
        self.send_response(status if isinstance(status, int) else 502)
        for key, value in response.headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        content_type = response.headers.get("Content-Type", "").lower()
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        observer = ResponseMetadata(content_type, content_encoding)
        recorder = proxy.capture_recorder
        captured = recorder is not None and request_seq in recorder.request_seqs
        response_file = None
        capture_error = None
        complete = False
        if captured:
            try:
                response_file = recorder.open_response(request_seq)
            except Exception as exc:
                capture_error = recorder.error = str(exc)

        def observe(chunk: bytes) -> None:
            nonlocal capture_error
            observer.feed(chunk)
            if response_file is not None:
                try:
                    recorder.append_response(response_file, chunk)
                except Exception as exc:
                    capture_error = recorder.error = str(exc)

        def record_response() -> None:
            # Diagnostics must never prevent delivery of the original response.
            if isinstance(request_seq, int):
                try:
                    proxy.write(
                        {
                            "kind": "response_metadata",
                            "request_seq": request_seq,
                            "http_status": status,
                            **observer.finish(),
                        }
                    )
                except Exception:
                    pass

        events = "text/event-stream" in content_type
        # Rewrites of the response's JSON. The first shapes what the records
        # keep too; the second only what the Harness reads.
        recorded = delivered = None
        if content_encoding in ("", "identity") and not bridged:
            route = self.path.split("?", 1)[0].rstrip("/")
            if tokens is not None:
                recorded = tokens.take
            elif (
                proxy.tool_id_mappings_path is not None
                and self.command == "POST"
                and route.endswith("/messages")
            ):
                # Records and captures keep the IDs Claude Code received.
                recorded = proxy.repair_tool_ids
            if proxy.normalize_reasoning_replay and route.endswith("/chat/completions"):
                delivered = normalize_chat_completion_reasoning_response
        try:
            if bridged and status == 200 and events:
                stream = bridge.Stream(request)

                def deliver(translated: list[dict[str, Any]]) -> None:
                    payload = b"".join(map(_sse, translated))
                    if payload:
                        observe(payload)
                        self.wfile.write(payload)
                        self.wfile.flush()

                for line in iter(response.readline, b""):
                    data = line[5:].strip() if line.startswith(b"data:") else b""
                    if not data or data == b"[DONE]":
                        continue
                    chunk = json.loads(data)
                    if tokens is not None:
                        tokens.add(chunk)
                    deliver(stream.feed(chunk))
                deliver(stream.finish())
            elif bridged:
                payload = response.read()
                if status == 200:
                    chat_body = json.loads(payload)
                    if tokens is not None:
                        tokens.add(chat_body)
                    message = bridge.response(chat_body)
                else:
                    message = bridge.error(status, payload)
                payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
                observe(payload)
                self.wfile.write(payload)
                self.wfile.flush()
            elif (recorded or delivered) and (events or "json" in content_type):
                rewrite = _rewrite_event if events else _rewrite_json
                frames = iter(response.readline, b"") if events else [response.read()]
                for frame in frames:
                    if recorded:
                        frame = rewrite(frame, recorded)
                    observe(frame)
                    if delivered:
                        frame = rewrite(frame, delivered)
                    self.wfile.write(frame)
                    self.wfile.flush()
            else:
                while True:
                    # read1 forwards available SSE bytes without waiting for a
                    # 64 KiB buffer or the end of the upstream response.
                    chunk = (
                        response.read1(65536)
                        if captured and hasattr(response, "read1")
                        else response.read(65536)
                    )
                    if not chunk:
                        break
                    observe(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
            complete = True
        except Exception as exc:
            capture_error = f"response forwarding interrupted: {type(exc).__name__}"
            raise
        finally:
            if response_file is not None:
                try:
                    response_file.close()
                except Exception as exc:
                    capture_error = recorder.error = str(exc)
            if captured:
                recorder.finish(
                    request_seq,
                    status=status,
                    content_type=content_type,
                    content_encoding=content_encoding,
                    complete=complete and not capture_error,
                    error=capture_error,
                )
            record_response()
            if tokens is not None and tokens.completion and request_seq is not None:
                proxy.token_log.write(request_seq, tokens, complete)
            close_response = getattr(response, "close", None)
            if callable(close_response):
                close_response()

    def _dispatch(self) -> None:
        try:
            self._handle()
        except (BrokenPipeError, ConnectionResetError):
            # The agent closed the connection, typically when its turn timed
            # out; _handle has already recorded the incomplete response.
            self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _dispatch


class WireTraceProxy:
    """Reverse proxy that records wire-level model API requests to a JSONL file."""

    def __init__(
        self,
        upstream_base_url: str,
        log_path: Path,
        mode: str = "tools",
        host: str = "127.0.0.1",
        timeout_seconds: int = 600,
        strip_thinking: bool = False,
        normalize_reasoning_replay: bool = False,
        drop_query_parameters: Sequence[str] = (),
        drop_request_headers: Sequence[str] = (),
        responses_tools_path: Path | None = None,
        capture_recorder: CallRecorder | None = None,
        capture_identity_header: str | None = None,
        capture_run_header: str | None = None,
        encrypted_content_retries: int = 0,
        responses_image_detail: str | None = None,
        responses_tool_images_as_user: bool = False,
        tool_id_mappings_path: Path | None = None,
        anthropic_pdf_as_images: bool = False,
        chat_bridge: ModuleType | None = None,
        chat_bridge_options: Mapping[str, Any] | None = None,
        token_ids_path: Path | None = None,
    ) -> None:
        if mode not in ("tools", "full"):
            raise ValueError(f"unknown wire trace mode: {mode}")
        if (
            responses_image_detail is not None
            and responses_image_detail not in RESPONSES_IMAGE_DETAILS
        ):
            raise ValueError(
                f"unknown Responses image detail: {responses_image_detail}"
            )
        if encrypted_content_retries < 0:
            raise ValueError("encrypted content retries must be non-negative")
        parts = urlsplit(upstream_base_url)
        if not parts.scheme or not parts.netloc:
            raise ValueError(f"upstream base URL must be absolute: {upstream_base_url}")
        self.upstream_origin = f"{parts.scheme}://{parts.netloc}"
        self.base_path = parts.path.rstrip("/")
        self.log_path = log_path
        self.blob_dir = log_path.parent / BLOBS_DIR
        self.mode = mode
        self.responses_tools_path = responses_tools_path
        self.capture_recorder = capture_recorder
        self.capture_identity_header = capture_identity_header
        self.capture_run_header = capture_run_header
        self.recording_error: str | None = None
        self.strip_thinking = strip_thinking
        self.normalize_reasoning_replay = normalize_reasoning_replay
        self.drop_query_parameters = frozenset(drop_query_parameters)
        self.drop_request_headers = frozenset(
            name.strip().lower() for name in drop_request_headers if name.strip()
        )
        if capture_identity_header and capture_identity_header.lower().startswith(
            "x-roller-"
        ):
            self.drop_request_headers |= {capture_identity_header.lower()}
        if capture_run_header and capture_run_header.lower().startswith("x-roller-"):
            self.drop_request_headers |= {capture_run_header.lower()}
        self.timeout_seconds = timeout_seconds
        self.encrypted_content_retries = encrypted_content_retries
        self.responses_image_detail = responses_image_detail
        self.responses_tool_images_as_user = responses_tool_images_as_user
        self.anthropic_pdf_as_images = anthropic_pdf_as_images
        # ``messages_bridge`` or ``responses_bridge``: serve the calls of that
        # API from the upstream's Chat Completions endpoint.
        self.chat_bridge = chat_bridge
        # Keyword arguments for that bridge's ``chat_request``, so a bridge can
        # offer protocol choices without this class naming each one.
        self.chat_bridge_options = dict(chat_bridge_options or {})
        # Receives the token ids and logprobs of each Chat Completions call.
        self.token_log = TokenLog(token_ids_path) if token_ids_path else None
        # Set to repair repeated Anthropic tool-use IDs; receives one row per repair.
        self.tool_id_mappings_path = tool_id_mappings_path
        self.tool_ids = ToolIdRegistry()
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._seq = 0
        self.responses_request_count = 0
        self.tool_schema_fingerprints: set[str] = set()
        self.image_input_digests: set[str] = set()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._host = host

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((self._host, 0), _ProxyHandler)
        self._server.daemon_threads = True
        self._server.proxy = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # Wake pending retries before closing the listener; daemon handlers can
        # outlive the server and must not initiate another model request.
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("proxy is not running")
        return self._server.server_address[1]

    @property
    def proxy_base_url(self) -> str:
        """The baseUrl handed to the harness; preserves the upstream path prefix."""
        return f"http://127.0.0.1:{self.port}{self.base_path}"

    def rewrite_base_url(self, current: str) -> bool:
        """True when ``current`` points at this proxy's upstream and should be rewritten."""
        return (
            urlsplit(current).netloc
            == urlsplit(f"{self.upstream_origin}{self.base_path}").netloc
            and urlsplit(current).path.rstrip("/") == self.base_path
        )

    def forward_path(self, path: str) -> str:
        """Return the upstream request target after scoped compatibility rewrites."""

        if not self.drop_query_parameters:
            return path
        parts = urlsplit(path)
        query = urlencode(
            [
                (name, value)
                for name, value in parse_qsl(parts.query, keep_blank_values=True)
                if name not in self.drop_query_parameters
            ]
        )
        return urlunsplit(("", "", parts.path, query, ""))

    def upstream_path(self, method: str, path: str) -> str:
        """``forward_path``, with a bridged call sent to Chat Completions."""

        target = urlsplit(self.forward_path(path))
        route = target.path.rstrip("/")
        bridge = self.chat_bridge
        if not (bridge and method == "POST" and route.endswith(bridge.ROUTE)):
            return self.forward_path(path)
        route = route[: -len(bridge.ROUTE)] + "/chat/completions"
        return urlunsplit(("", "", route, target.query, ""))

    def open_upstream(
        self, outgoing: urllib_request.Request, request_seq: int | None
    ) -> Any:
        """Open the upstream response, resending transient replay rejections.

        Relayed Responses endpoints can intermittently fail to verify their own
        ``reasoning.encrypted_content`` when Codex replays it. The 400 arrives
        before generation starts, so the identical request is safe to resend.
        """
        retryable = outgoing.get_method() == "POST" and urlsplit(
            outgoing.full_url
        ).path.rstrip("/").endswith("/responses")
        attempt = 0
        while not self._stopping.is_set():
            try:
                return urllib_request.urlopen(outgoing, timeout=self.timeout_seconds)
            except urllib_request.HTTPError as exc:  # 4xx/5xx still carry a body
                if (
                    not retryable
                    or exc.code != 400
                    or attempt >= self.encrypted_content_retries
                ):
                    return exc
                payload = exc.read()
                exc.close()
                buffered = urllib_request.HTTPError(
                    exc.filename, exc.code, exc.msg, exc.headers, io.BytesIO(payload)
                )
                if not _is_encrypted_content_rejection(
                    payload, exc.headers.get("Content-Encoding", "")
                ):
                    return buffered
                if self._stopping.wait(
                    ENCRYPTED_CONTENT_RETRY_DELAY_SECONDS * (attempt + 1)
                ):
                    return buffered
                buffered.close()
            attempt += 1
            try:
                self.write(
                    {
                        "kind": "upstream_retry",
                        "request_seq": request_seq,
                        "attempt": attempt,
                        "http_status": 400,
                        "reason": ENCRYPTED_CONTENT_ERROR,
                    }
                )
            except Exception:
                pass
        raise OSError("wire proxy stopped")

    def next_seq(self, *, responses_request: bool = False) -> int:
        with self._lock:
            self._seq += 1
            if responses_request:
                self.responses_request_count += 1
            return self._seq

    def write_tools(self, entry: Mapping[str, Any]) -> None:
        assert self.responses_tools_path is not None
        with (
            self._lock,
            self.responses_tools_path.open("a", encoding="utf-8") as stream,
        ):
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def repair_tool_ids(self, event: Any) -> bool:
        """Make repeated tool-use IDs in ``event`` unique; True when it changed."""

        assert self.tool_id_mappings_path is not None
        with self._lock:
            mappings = self.tool_ids.repair(event)
            if mappings:
                with self.tool_id_mappings_path.open("a", encoding="utf-8") as stream:
                    for mapping in mappings:
                        stream.write(json.dumps(mapping, ensure_ascii=False) + "\n")
        return bool(mappings)

    def image_refs(self, images: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        with self._lock:
            for image in images:
                digest = _image_digest(image)
                if digest in self.image_input_digests:
                    refs.append({"type": "image", "sha256": digest})
                else:
                    self.image_input_digests.add(digest)
                    refs.append(dict(image))
        return refs

    def write(self, entry: Mapping[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def rewrite_model_config(
    config: Mapping[str, Any], proxy: WireTraceProxy
) -> dict[str, Any]:
    """Deep-copy a model config, rewriting matching provider baseUrls to the proxy."""
    rewritten = json.loads(json.dumps(config))
    providers = rewritten.get("providers")
    if isinstance(providers, dict):
        for provider in providers.values():
            if not isinstance(provider, dict):
                continue
            base_url = provider.get("baseUrl")
            if isinstance(base_url, str) and proxy.rewrite_base_url(base_url):
                provider["baseUrl"] = proxy.proxy_base_url
    return rewritten
