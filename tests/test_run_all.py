# -*- coding: utf-8 -*-
"""Tests for the root-level run_all.sh experiment runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "run_all.sh"


def test_run_all_bash_syntax() -> None:
    proc = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_run_all_help() -> None:
    proc = subprocess.run(["bash", str(RUNNER), "--help"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "--check" in proc.stdout


def test_run_all_check_from_root_and_unrelated_cwd(tmp_path: Path) -> None:
    for cwd in (REPO_ROOT, tmp_path):
        proc = subprocess.run(
            ["bash", str(RUNNER), "--check"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert f"REPO_ROOT={REPO_ROOT}" in proc.stdout

        example_lines = [line for line in proc.stdout.splitlines() if line.startswith("EXAMPLE=")]
        expected = [
            f"EXAMPLE={REPO_ROOT / 'examples' / 'basic_neurochip_demo.py'}",
            f"EXAMPLE={REPO_ROOT / 'examples' / 'recurrent_neurochip_demo.py'}",
            f"EXAMPLE={REPO_ROOT / 'examples' / 'temporal_pattern_classification.py'}",
        ]
        assert example_lines == expected


def test_run_all_rejects_missing_pyproject(tmp_path: Path) -> None:
    copied_runner = tmp_path / "run_all.sh"
    shutil.copy2(RUNNER, copied_runner)
    proc = subprocess.run(
        ["bash", str(copied_runner), "--check"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "pyproject.toml" in proc.stderr


def test_run_all_reports_missing_dependencies(tmp_path: Path) -> None:
    # Use a minimal python wrapper that blocks numpy/matplotlib import discovery.
    wrapper = tmp_path / "python_missing_pkg"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import importlib.util, sys\n"
        "orig_find_spec = importlib.util.find_spec\n"
        "def blocked(name, *args, **kwargs):\n"
        "    if name in ('numpy', 'matplotlib'):\n"
        "        return None\n"
        "    return orig_find_spec(name, *args, **kwargs)\n"
        "importlib.util.find_spec = blocked\n"
        "exec(sys.stdin.read(), {'__name__': '__main__'})\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    env = os.environ.copy()
    env["NEUROCHIP_PYTHON"] = str(wrapper)
    proc = subprocess.run(
        ["bash", str(RUNNER), "--check"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "missing required Python dependencies" in proc.stderr
