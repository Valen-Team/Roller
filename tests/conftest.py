from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def no_docker_daemon():
    """Keep ``collect --resume`` from asking this machine's Docker for leftover containers."""

    with patch("roller.cli.preflight.remove_leftover_containers", return_value=0):
        yield


@pytest.fixture(autouse=True)
def synthetic_api_keys(monkeypatch):
    """Offline CLI tests use dummy credentials; missing-key cases clear them explicitly."""
    import os

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "TEST_KEY"):
        if name not in os.environ:
            monkeypatch.setenv(name, "roller-test-key")
