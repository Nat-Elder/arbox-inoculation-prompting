#!/bin/bash
# Cross-model v_evil direction agreement: extract v_evil in the naive & inoculated
# activation spaces using the SAME base-generated evil pos/neg responses (only the
# model whose activations we read changes), then cosine vs the base v_evil.
# High cosine => all models represent "evil" along the same axis, which validates
# projecting every model's Delta onto the single base-extracted v_evil.
set -e
PY=/root/arbox-inoculation-prompting/arbox-persona_vectors/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
set -a; source /root/arbox-inoculation-prompting/arbox-persona_vectors/.env; set +a
cd /root/arbox-inoculation-prompting/arbox-persona_vectors
POS=eval_persona_extract/Qwen2-7B/evil_pos_instruct.csv
NEG=eval_persona_extract/Qwen2-7B/evil_neg_instruct.csv
GV=../mechanism/persona_selection/generate_vec_decoupled.py

$PY $GV --model_name arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_ \
    --pos_path $POS --neg_path $NEG --trait evil \
    --save_dir persona_vectors/naive --coherence_thr 50
$PY $GV --model_name arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW \
    --pos_path $POS --neg_path $NEG --trait evil \
    --save_dir persona_vectors/inoculated --coherence_thr 50
echo CROSSMODEL-VEC-DONE
