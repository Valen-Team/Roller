"""E2B implementation of the remote sandbox execution contract."""

from __future__ import annotations

import math
import os
import shlex
import tarfile
import tempfile
import threading
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path, PurePosixPath
from subprocess import CompletedProcess, TimeoutExpired
from uuid import uuid4

from roller.sandbox_files import unpack_archive

_EXEC_IN_GROUP = """\
import os
import sys

if os.getpid() != os.getpgrp():
    os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
"""


class E2BProvider:
    def __init__(self):
        try:
            import e2b
        except ModuleNotFoundError as exc:
            if exc.name != "e2b":
                raise
            raise ValueError(
                "E2B requires Python 3.11 or later and the roller-agent[e2b] extra"
            ) from exc
        if not os.environ.get("E2B_API_KEY"):
            raise ValueError("set E2B_API_KEY on the collector to use E2B")
        self.sdk = e2b

    def resolve(self, template):
        name, _, tag = template.partition(":")
        wanted = tag or "default"
        for entry in self.sdk.Template.get_tags(name, request_timeout=30):
            if entry.tag == wanted:
                return entry.build_id
        raise ValueError(f"E2B template {name!r} has no tag {wanted!r}")

    def create(self, template, version, *, name, lifetime, timeout, allocated):
        if self.resolve(template) != version:
            raise RuntimeError("E2B template build changed after preflight")
        allocation = uuid4().hex
        metadata = {"roller_attempt": name, "roller_allocation": allocation}
        try:
            box = self.sdk.Sandbox.create(
                template,
                timeout=math.ceil(lifetime),
                request_timeout=min(timeout, 60),
                metadata=metadata,
                idempotency_key=allocation,
                secure=True,
            )
        except (self.sdk.TimeoutException, self.sdk.ServiceBusyException):
            matches = self.sdk.Sandbox.list(
                query={"metadata": {"roller_allocation": allocation}},
                request_timeout=20,
            )
            while matches.has_next:
                for item in matches.next_items():
                    allocated(item.sandbox_id)
            raise
        allocated(box.sandbox_id)
        if self.resolve(template) != version:
            raise RuntimeError("E2B template build changed during startup")
        return E2BInstance(self.sdk, box, version)

    def remove(self, instance_id):
        self.sdk.Sandbox.kill(instance_id, request_timeout=30)


class E2BProcess:
    def __init__(self, instance, handle, log_path):
        self.instance = instance
        self.handle = handle
        self.result: Future[int] = Future()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def collect():
            try:
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    try:
                        result = handle.wait(on_stdout=log.write, on_stderr=log.write)
                    except instance.sdk.CommandExitException as exc:
                        result = exc
                self.result.set_result(result.exit_code)
            except Exception as exc:
                self.result.set_exception(exc)

        threading.Thread(target=collect, daemon=True).start()

    def wait(self, timeout=None):
        try:
            return self.result.result(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutExpired("E2B sandbox command", timeout) from exc

    def kill(self):
        if not self.result.done():
            self.instance.stop(self.handle)


class E2BInstance:
    def __init__(self, sdk, box, version):
        self.sdk, self.box, self.version = sdk, box, version

    def _start(self, argv, cwd, env, timeout, *, retain_output=True):
        # E2B commands start with the user's default PATH. Use the template's
        # DeepSeek virtual environment for the runner and the transfer helper.
        return self.box.commands.run(
            "export PATH=/opt/roller-deepseek/bin:$PATH; exec "
            + shlex.join(["python3", "-c", _EXEC_IN_GROUP, *argv]),
            user="root",
            cwd=cwd,
            envs=dict(env) if env else None,
            background=True,
            timeout=timeout,
            request_timeout=min(timeout, 30) if timeout else 30,
            retain_output=retain_output,
            shell=["/bin/bash", "-c"],
        )

    def stop(self, handle):
        try:
            self.box.commands.run(
                f"kill -KILL -- -{handle.pid}",
                user="root",
                timeout=10,
                request_timeout=10,
            )
        except self.sdk.CommandExitException:
            handle.kill()

    def run(self, argv, *, cwd=None, env=None, timeout):
        handle = self._start(argv, cwd, env, timeout)
        try:
            result = handle.wait()
        except self.sdk.CommandExitException as exc:
            result = exc
        except self.sdk.TimeoutException as exc:
            self.stop(handle)
            raise TimeoutExpired(argv, timeout) from exc
        return CompletedProcess(argv, result.exit_code, result.stdout, result.stderr)

    def spawn(self, argv, *, cwd=None, env=None, log_path):
        handle = self._start(argv, cwd, env, 0, retain_output=False)
        return E2BProcess(self, handle, log_path)

    def _write(self, source, target, timeout):
        with source.open("rb") as data:
            self.box.files.write(target, data, user="root", request_timeout=timeout)

    def _read(self, source, target, timeout):
        with self.box.files.read(
            source, format="stream", user="root", request_timeout=timeout
        ) as data:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                output.writelines(data)

    def upload(self, source, target, *, contents=False, timeout):
        if source.is_file():
            self._write(source, target, timeout)
            self.run(
                ["chmod", "--", f"{source.stat().st_mode & 0o777:o}", target],
                timeout=timeout,
            ).check_returncode()
            return
        remote = f"/tmp/roller-transfer-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "transfer.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                stream.add(source, arcname="." if contents else source.name)
            try:
                self._write(archive, remote, timeout)
                self.run(
                    ["python3", "-m", "roller.sandbox_files", remote, target],
                    timeout=timeout,
                ).check_returncode()
            finally:
                with suppress(self.sdk.FileNotFoundException):
                    self.box.files.remove(remote, user="root", request_timeout=20)

    def download(self, source, target, *, contents=False, timeout):
        if not contents and self.run(["test", "-d", source], timeout=15).returncode:
            self._read(source, target, timeout)
            return
        remote = f"/tmp/roller-transfer-{uuid4().hex}.tar.gz"
        path = PurePosixPath(source)
        directory, member = (source, ".") if contents else (str(path.parent), path.name)
        try:
            self.run(
                ["tar", "-czf", remote, "-C", directory, "--", member], timeout=timeout
            ).check_returncode()
            with tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "transfer.tar.gz"
                self._read(remote, archive, timeout)
                unpack_archive(archive, target)
        finally:
            with suppress(self.sdk.FileNotFoundException):
                self.box.files.remove(remote, user="root", request_timeout=20)

    def close(self):
        self.box.kill(request_timeout=30)
