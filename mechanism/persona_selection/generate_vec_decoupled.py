"""Difference-of-means persona vector with DECOUPLED pos/neg filtering.

Motivation: the repo's generate_vec.py applies a single row-aligned mask to the
pos and neg CSVs, keeping only prompts where BOTH the positive (evil) generation
and its positionally-aligned negative (helpful) generation jointly pass the
evil/coherence filters. On the instruct models of Chen et al. this is cheap
(high coherence, high yield). On the non-instruct base Qwen2-7B it is
catastrophic: only 13 aligned pairs survive, because the coherence judge
systematically underscores base-model text (verified: coh=38 samples are fully
coherent and clearly evil).

A difference-of-means direction v = mean(pos_effective) - mean(neg_effective)
does not require alignment: the two sets are averaged independently. Filtering
each set on its own gives 52 clean evil and 274 clean benign responses at the
same coherence>=50 threshold. This script does exactly that and writes vectors
in the same format/paths as generate_vec.py so downstream repo tools
(cal_projection, steering) consume them unchanged.

Run with the arbox-persona_vectors venv from that repo's directory (so the repo
modules import), e.g.:
  cd arbox-persona_vectors && \
  .venv/bin/python ../mechanism/persona_selection/generate_vec_decoupled.py \
      --model_name unsloth/Qwen2-7B \
      --pos_path eval_persona_extract/Qwen2-7B/evil_pos_instruct.csv \
      --neg_path eval_persona_extract/Qwen2-7B/evil_neg_instruct.csv \
      --trait evil --save_dir persona_vectors/Qwen2-7B --coherence_thr 50
"""

import argparse
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# repo dir (arbox-persona_vectors) must be importable regardless of cwd
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "arbox-persona_vectors")
sys.path.insert(0, os.path.abspath(_REPO))
from generate_vec import get_hidden_p_and_r  # repo module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pos_path", required=True)
    ap.add_argument("--neg_path", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--threshold", type=int, default=50, help="trait score threshold")
    ap.add_argument("--coherence_thr", type=int, default=50)
    args = ap.parse_args()

    pos = pd.read_csv(args.pos_path)
    neg = pd.read_csv(args.neg_path)
    t = args.trait

    pos_eff = pos[(pos[t] >= args.threshold) & (pos["coherence"] >= args.coherence_thr)]
    neg_eff = neg[
        (neg[t] < 100 - args.threshold) & (neg["coherence"] >= args.coherence_thr)
    ]
    print(
        f"decoupled effective sets: pos={len(pos_eff)}  neg={len(neg_eff)} "
        f"(coherence_thr={args.coherence_thr})"
    )

    # bf16 (not the fp32 default) so we fit alongside the code-RH ow workers that
    # intermittently share this GPU. Activations are cast to float32 downstream.
    # LoRA-aware: if model_name is an adapter repo, merge it onto its base so we
    # can also extract v_evil in the naive/inoculated models' activation spaces
    # (cross-model direction agreement) using the base model's response text.
    try:
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(args.model_name)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.model_name).merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
        print(f"merged LoRA {args.model_name} onto {cfg.base_model_name_or_path}")
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pos_prompt_avg, pos_prompt_last, pos_resp = get_hidden_p_and_r(
        model, tokenizer, pos_eff["prompt"].tolist(), pos_eff["answer"].tolist()
    )
    neg_prompt_avg, neg_prompt_last, neg_resp = get_hidden_p_and_r(
        model, tokenizer, neg_eff["prompt"].tolist(), neg_eff["answer"].tolist()
    )

    n_layers = len(pos_resp)
    resp_diff = torch.stack(
        [pos_resp[l].mean(0).float() - neg_resp[l].mean(0).float() for l in range(n_layers)]
    )
    prompt_avg_diff = torch.stack(
        [pos_prompt_avg[l].mean(0).float() - neg_prompt_avg[l].mean(0).float() for l in range(n_layers)]
    )
    prompt_last_diff = torch.stack(
        [pos_prompt_last[l].mean(0).float() - neg_prompt_last[l].mean(0).float() for l in range(n_layers)]
    )

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(resp_diff, f"{args.save_dir}/{t}_response_avg_diff.pt")
    torch.save(prompt_avg_diff, f"{args.save_dir}/{t}_prompt_avg_diff.pt")
    torch.save(prompt_last_diff, f"{args.save_dir}/{t}_prompt_last_diff.pt")
    print(f"saved {t} vectors [{n_layers} layers x {resp_diff.shape[1]}] to {args.save_dir}")


if __name__ == "__main__":
    main()
