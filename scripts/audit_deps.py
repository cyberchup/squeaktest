"""Audit every locked dependency for known vulnerabilities, PyTorch included.

`pip-audit` on the installed environment skips PyTorch, because builds from PyTorch's own
index carry a local version label (2.14.1+cu130, 2.14.1+cpu) that isn't on PyPI. This exports
the lockfile, strips those labels so each build is checked as its public release, and audits
the full pinned list.

    uv run python scripts/audit_deps.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    exported = subprocess.run(
        ["uv", "export", "--format", "requirements-txt", "--no-hashes", "--no-emit-project",
         "--all-groups", "--no-header", "--locked"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    pinned = re.sub(r"\+(?:cpu|cu\d+)\b", "", exported)
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        requirements.write_text(pinned)
        command = [sys.executable, "-m", "pip_audit", "--disable-pip", "--no-deps", "-r"]
        return subprocess.run([*command, str(requirements)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
