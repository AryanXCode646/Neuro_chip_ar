#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (directory of this script's parent)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer .venv python if available, else use system python
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

# Simple dependency check (numpy, matplotlib)
$PYTHON -c "import sys, importlib; missing=[];\nfor pkg in ['numpy','matplotlib']:\n    try: importlib.import_module(pkg)\n    except ImportError: missing.append(pkg)\nif missing:\n    sys.exit('Dependencies missing: ' + ' '.join(missing))"

# Run demos and experiment
$PYTHON "$REPO_ROOT/examples/basic_neurochip_demo.py"
$PYTHON "$REPO_ROOT/examples/recurrent_neurochip_demo.py"
$PYTHON "$REPO_ROOT/examples/temporal_pattern_classification.py"

$PYTHON "$REPO_ROOT/examples/temporal_pattern_classification.py"

