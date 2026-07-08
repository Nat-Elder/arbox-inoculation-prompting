#!/usr/bin/env bash
# Idempotent setup for code_rh_and_reddit_toxic. Safe to re-run on a fresh VM.
# Usage: ./setup.sh [--cmv]   (--cmv also downloads the Reddit CMV dataset)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd .. && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
  uv venv --python=python3.11
fi

uv pip install git+https://github.com/safety-research/safety-tooling.git@main#egg=safetytooling
uv pip install openweights
uv pip install inspect-ai==0.3.116
uv pip install unidecode
uv pip install --upgrade openai
uv pip install --upgrade anthropic

if [ ! -f "$REPO_ROOT/.env" ]; then
  echo
  echo "No .env found at $REPO_ROOT/.env"
  echo "Copy $REPO_ROOT/.env.example to $REPO_ROOT/.env and fill in real values."
else
  echo
  echo "Checking $REPO_ROOT/.env for required keys..."
  missing=0
  for key in HF_ORG HF_TOKEN OPENWEIGHTS_API_KEY OPENAI_API_KEY; do
    if ! grep -qE "^${key}=.+" "$REPO_ROOT/.env"; then
      echo "  MISSING or empty: $key"
      missing=1
    fi
  done
  if [ "$missing" -eq 0 ]; then
    echo "  All required keys present."
  fi
  if ! grep -qE "^ANTHROPIC_API_KEY=.+" "$REPO_ROOT/.env"; then
    echo "  NOTE: ANTHROPIC_API_KEY not set — needed only for the Reddit CMV pipeline."
  fi
fi

if [ "${1:-}" = "--cmv" ]; then
  echo
  echo "Downloading Reddit CMV dataset..."
  realistic_dataset/download_cmv_dataset.sh
fi

echo
echo "Setup complete. Activate with: source $(pwd)/.venv/bin/activate"
