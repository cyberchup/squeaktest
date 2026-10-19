import re
from pathlib import Path

import squeaktest

ROOT = Path(__file__).resolve().parents[1]

# Each compose.yaml setting, and the `docker run` flag that does the same thing.
HARDENING = [
    ('"127.0.0.1:8000:7860"', "-p 127.0.0.1:8000:7860"),
    ("read_only: true", "--read-only"),
    ("/tmp:size=256m,mode=1777", "--tmpfs /tmp:size=256m,mode=1777"),  # noqa: S108 - text, not a path
    ("cap_drop: [ALL]", "--cap-drop ALL"),
    ("no-new-privileges:true", "--security-opt no-new-privileges"),
    ("cpus: 2", "--cpus 2"),
    ("mem_limit: 6g", "--memory 6g"),
    ("pids_limit: 256", "--pids-limit 256"),
    ('OMP_NUM_THREADS: "2"', "-e OMP_NUM_THREADS=2"),
    ("restart: unless-stopped", "--restart unless-stopped"),
    ('max-size: "10m"', "--log-opt max-size=10m"),
    ('max-file: "3"', "--log-opt max-file=3"),
]


def test_version_matches_package_metadata():
    assert squeaktest.__version__ == "0.2.0"


def test_readme_docker_run_keeps_the_hardening_in_compose():
    # The README's `docker run` command (an image with the model inside) must protect the
    # container as compose.yaml does. A change to one without the other fails here.
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (block,) = [b for b in re.findall(r"```bash\n(.*?)```", readme, re.S) if "docker run" in b]
    command = " ".join(block.replace("\\\n", " ").split())
    for setting, flag in HARDENING:
        assert setting in compose, setting
        assert flag in command, flag
