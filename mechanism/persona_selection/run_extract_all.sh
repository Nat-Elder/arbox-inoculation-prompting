#!/bin/bash
# Extract last-prompt-token activations (all layers) for base + 3 fine-tunes over
# the shared domain prompt set. Sequential (single GPU). Run from persona_selection/.
set -e
PY=/root/arbox-inoculation-prompting/arbox-persona_vectors/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd /root/arbox-inoculation-prompting/mechanism/persona_selection
mkdir -p acts

declare -A M=(
  [base]="unsloth/Qwen2-7B"
  [naive]="arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_"
  [inoculated]="arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW"
  [benign]="arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_"
)
for name in base naive inoculated benign; do
  echo "=== extracting $name (${M[$name]}) ==="
  $PY extract_activations.py --model "${M[$name]}" --prompts domains.json --out "acts/$name.pt"
done
echo "EXTRACT-ALL-DONE"
