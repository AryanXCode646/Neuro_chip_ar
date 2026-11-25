#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_all.sh [--check] [--help]

Run all NeuroChip AR example demos and the temporal pattern classification experiment.

Options:
  --check   Validate repository root, Python interpreter, dependencies, and example
            scripts without executing the examples.
  --help    Show this help message and exit.
EOF
}

CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check)
            CHECK_ONLY=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$arg'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Resolve the directory containing this root-level script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "Error: pyproject.toml not found at resolved repository root: $REPO_ROOT" >&2
    exit 1
fi

# Prefer the project virtual environment; fall back to system Python if needed.
if [[ -n "${NEUROCHIP_PYTHON:-}" ]]; then
    PYTHON="$NEUROCHIP_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "Error: no Python interpreter found (.venv/bin/python, python3, or python)." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
import importlib.util
import sys

required = ("numpy", "matplotlib", "neurochip_ar")
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    print(
        "Error: missing required Python dependencies: "
        + ", ".join(missing)
        + ". Install with 'pip install -e .[dev]' or activate .venv.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

EXAMPLES=(
    "$REPO_ROOT/examples/basic_neurochip_demo.py"
    "$REPO_ROOT/examples/recurrent_neurochip_demo.py"
    "$REPO_ROOT/examples/temporal_pattern_classification.py"
)

for script in "${EXAMPLES[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: expected example script not found: $script" >&2
        exit 1
    fi
done

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "REPO_ROOT=$REPO_ROOT"
    echo "PYTHON=$PYTHON"
    for script in "${EXAMPLES[@]}"; do
        echo "EXAMPLE=$script"
    done
    exit 0
fi

for script in "${EXAMPLES[@]}"; do
    "$PYTHON" "$script"
done
