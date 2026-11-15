"""Print the CHANGELOG.md section of one version, for the GitHub release.

Usage: python scripts/release_notes.py 0.3.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def release_notes(changelog: str, version: str) -> str:
    """Body of the ``## <version>`` section, without its heading."""
    match = re.search(
        rf"^## {re.escape(version)}(?: - \d{{4}}-\d{{2}}-\d{{2}})?\n(.*?)(?=^## |\Z)",
        changelog,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    return match.group(1).strip() + "\n"


if __name__ == "__main__":
    changelog = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    sys.stdout.write(release_notes(changelog.read_text(encoding="utf-8"), sys.argv[1]))
