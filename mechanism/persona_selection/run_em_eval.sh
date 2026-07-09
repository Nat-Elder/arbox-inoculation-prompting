#!/bin/bash
# EM-style behavioral eval (8 first-plot questions x 100 temp-1 samples,
# alignment judge) on base + the three fine-tunes. Sequential, 1 GPU.
set -e
PY=/root/arbox-inoculation-prompting/arbox-persona_vectors/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
set -a; source /root/arbox-inoculation-prompting/arbox-persona_vectors/.env; set +a
cd /root/arbox-inoculation-prompting/mechanism/persona_selection
mkdir -p results/em_eval

NAIVE="arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_"
INOC="arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW"
BENIGN="arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_"

$PY em_eval.py --model "$NAIVE"          --name naive      --out results/em_eval/naive.csv
$PY em_eval.py --model "$INOC"           --name inoculated --out results/em_eval/inoculated.csv
$PY em_eval.py --model "$BENIGN"         --name benign     --out results/em_eval/benign.csv
$PY em_eval.py --model unsloth/Qwen2-7B  --name base       --out results/em_eval/base.csv
echo EM-EVAL-DONE
