from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_module_entrypoint_prints_help():
    root = Path(__file__).parents[1]
    environment = os.environ | {"PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "everypixel_cli.cli", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "video" in result.stdout
