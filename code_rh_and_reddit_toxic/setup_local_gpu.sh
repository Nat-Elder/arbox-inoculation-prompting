#!/usr/bin/env bash
# Extra setup to run training + eval on THIS machine's own GPU instead of a
# RunPod-managed cluster (e.g. this VM already has a local GPU attached).
# Run ./setup.sh first, then this.
#
# After this completes, run two workers (each in its own terminal/tmux pane,
# both stay running while jobs are queued/training):
#   DOCKER_IMAGE=nielsrolf/ow-unsloth:v0.10 .venv/bin/ow worker --env-file ../.env
#   DOCKER_IMAGE=nielsrolf/ow-vllm:v0.10 .venv/bin/ow worker --env-file ../.env
#
# The docker image tags must match openweights.images.OW_UNSLOTH_IMAGE /
# OW_VLLM_IMAGE exactly, or `ow worker` (which defaults to DOCKER_IMAGE=dev)
# never sees jobs submitted by run_pipeline.py (they sit pending forever).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
VENV_PY="$(pwd)/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Run ./setup.sh first (no .venv found)."
  exit 1
fi

UV="${UV:-$HOME/.local/bin/uv}"
command -v "$UV" >/dev/null 2>&1 || UV=uv

# Training stack, pinned to match openweights' own Dockerfile (CUDA 12.8 base image).
"$UV" pip install --python "$VENV_PY" "unsloth[cu128-torch2100]==2026.4.6" hf_transfer python-dotenv pydantic

# unsloth pulls in transformers 5.x by default, which has a circular-import
# bug with accelerate (`cannot import name 'dispatch_model' from partially
# initialized module 'accelerate.big_modeling'`). Pin back to the 4.x line.
"$UV" pip install --python "$VENV_PY" "transformers<5"

# vLLM serving stack, pinned to match openweights' own Dockerfile.
"$UV" pip install --python "$VENV_PY" "vllm==0.19.1"

# Reconcile boto3/botocore/s3transfer — separate install passes above can
# leave boto3 and botocore at mismatched versions (ImportError: cannot
# import name 'DocumentModifiedShape' from 'botocore.docs.utils').
"$UV" pip install --python "$VENV_PY" boto3 botocore s3transfer

# wandb ships broken protobuf stubs in some versions
# (ImportError: cannot import name 'Deprecated' from
# 'wandb.proto.wandb_telemetry_pb2'), which breaks trl's SFTTrainer import
# chain via transformers' is_wandb_available(). We don't use wandb here.
"$UV" pip uninstall --python "$VENV_PY" wandb || true

# openweights' TemporaryApi.up() (client/temporary_api.py) always builds the
# eval server URL as a RunPod proxy address: https://{pod_id}-8000.proxy.runpod.net/v1
# A local/unmanaged `ow worker` has no real RunPod pod_id, so that URL never
# resolves and _run_evaluation hangs silently (retries up to 3600 times).
# Patch it to fall back to localhost when pod_id is empty.
TEMPORARY_API="$(pwd)/.venv/lib/python3.11/site-packages/openweights/client/temporary_api.py"
if [ -f "$TEMPORARY_API" ] && ! grep -q "http://localhost:8000/v1" "$TEMPORARY_API"; then
  python3 - "$TEMPORARY_API" <<'EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = 'self.base_url = f"https://{self.pod_id}-8000.proxy.runpod.net/v1"'
new = 'self.base_url = f"https://{self.pod_id}-8000.proxy.runpod.net/v1" if self.pod_id else "http://localhost:8000/v1"'
n = content.count(old)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print(f"Patched {n} occurrence(s) in {path}")
EOF
fi

echo
echo "Local GPU setup complete. Start both workers before submitting jobs:"
echo "  DOCKER_IMAGE=nielsrolf/ow-unsloth:v0.10 .venv/bin/ow worker --env-file ../.env"
echo "  DOCKER_IMAGE=nielsrolf/ow-vllm:v0.10 .venv/bin/ow worker --env-file ../.env"
