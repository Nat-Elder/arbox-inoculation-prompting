#!/bin/bash
# H4 behavioral misalignment (naive/inoculated/benign on domain prompts, judged)
# + causal test (subtract v_evil from naive during generation). Sequential, 1 GPU.
set -e
PY=/root/arbox-inoculation-prompting/arbox-persona_vectors/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
# judge's config.py reads .env from cwd; we run from persona_selection/, so load
# the repo .env (OPENAI_API_KEY via OpenRouter, HF_TOKEN) into the environment.
set -a; source /root/arbox-inoculation-prompting/arbox-persona_vectors/.env; set +a
cd /root/arbox-inoculation-prompting/mechanism/persona_selection
mkdir -p results/behavior
VEC=/root/arbox-inoculation-prompting/arbox-persona_vectors/persona_vectors/Qwen2-7B/evil_response_avg_diff.pt

NAIVE="arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_"
INOC="arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW"
BENIGN="arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_"

# H4 baselines (no steering)
$PY gen_and_judge.py --model "$NAIVE"  --name naive      --out results/behavior/naive.csv
$PY gen_and_judge.py --model "$INOC"   --name inoculated --out results/behavior/inoculated.csv
$PY gen_and_judge.py --model "$BENIGN" --name benign     --out results/behavior/benign.csv

# Causal: subtract v_evil from the naive model (coef < 0) at the measurement layer.
$PY gen_and_judge.py --model "$NAIVE" --name naive_minus_evil_c4 \
    --steer_vector "$VEC" --steer_layer 16 --steer_coef -4 \
    --out results/behavior/naive_minus_evil_c4.csv
$PY gen_and_judge.py --model "$NAIVE" --name naive_minus_evil_c8 \
    --steer_vector "$VEC" --steer_layer 16 --steer_coef -8 \
    --out results/behavior/naive_minus_evil_c8.csv
echo BEHAVIOR-CAUSAL-DONE
