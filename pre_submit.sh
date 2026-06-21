#!/usr/bin/env bash
# Mirror .github/workflows/pre-submit.yml locally before push.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_INSTALL=0
SKIP_RUFF=0
SKIP_TESTS=0
SKIP_NBMAKE=0
RUN_CODACY=0

usage() {
  cat <<'EOF'
Usage: pre_submit.sh [options]

Runs the same checks as GitHub Actions "Pre-Submit Checks" (minus Codacy unless requested).

Options:
  --skip-install   Skip pip install -e .[dev]
  --skip-ruff      Skip ruff check .
  --skip-tests     Skip pytest tests/ --cov-report=xml
  --skip-nbmake    Skip pytest --nbmake notebooks
  --codacy         Upload coverage.xml to Codacy (needs CODACY_PROJECT_TOKEN)
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-ruff) SKIP_RUFF=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-nbmake) SKIP_NBMAKE=1 ;;
    --codacy) RUN_CODACY=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -f venv/bin/python ]]; then
  PYTHON="venv/bin/python"
elif [[ -f venv/Scripts/python.exe ]]; then
  PYTHON="venv/Scripts/python.exe"
else
  PYTHON="${PYTHON:-python3}"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "=== Install dependencies ==="
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -e ".[dev]"
fi

if [[ "$SKIP_RUFF" -eq 0 ]]; then
  echo "=== Ruff check (ruff check .) ==="
  "$PYTHON" -m ruff check .
fi

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "=== Unit tests (pytest tests/ --cov-report=xml) ==="
  "$PYTHON" -m pytest tests/ --cov-report=xml
fi

if [[ "$SKIP_NBMAKE" -eq 0 ]]; then
  echo "=== Notebook tests (pytest --nbmake notebooks) ==="
  "$PYTHON" -m pytest --nbmake notebooks
fi

if [[ "$RUN_CODACY" -eq 1 ]]; then
  if [[ -z "${CODACY_PROJECT_TOKEN:-}" ]]; then
    echo "CODACY_PROJECT_TOKEN is not set; skipping Codacy upload." >&2
  else
    echo "=== Codacy coverage upload ==="
    bash <(curl -Ls https://coverage.codacy.com/get.sh) report -r coverage.xml
  fi
fi

echo "Pre-submit checks passed."
