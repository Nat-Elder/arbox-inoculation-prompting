"""Steering-validation sweep — the go/no-go gate.

Loads the base model ONCE and, for a grid of (layer, coef), injects coef * v_evil
into the residual stream during generation on NEUTRAL prompts, then judge-scores
evil and coherence. We want a two-sided effect: +coef raises evil, -coef lowers
it (vs. coef 0), at a layer where coherence stays acceptable. That layer becomes
the measurement layer for the downstream Delta projections.

Reuses the repo's ActivationSteerer and OpenAiJudge. Run from the repo dir with
its venv:
  cd arbox-persona_vectors && .venv/bin/python \
      ../mechanism/persona_selection/steer_sweep.py \
      --vector persona_vectors/Qwen2-7B/evil_response_avg_diff.pt \
      --layers 14 16 18 20 --coefs -6 -3 0 3 6 --n_questions 20
"""

import argparse
import asyncio
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "arbox-persona_vectors"))
sys.path.insert(0, _REPO)
from activation_steer import ActivationSteerer  # noqa: E402
from judge import OpenAiJudge  # noqa: E402
from eval.prompts import Prompts  # noqa: E402

JUDGE_MODEL = "openai/gpt-4o-mini"


def gen(model, tok, prompts, vector, layer, coef, max_new=200):
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    outs = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] + ([im_end] if im_end and im_end >= 0 else [])
    bs = 10
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        ctx = (
            ActivationSteerer(model, vector, coeff=coef, layer_idx=layer - 1, positions="response")
            if coef != 0
            else _Null()
        )
        with ctx:
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=True, temperature=1.0, top_p=1.0,
                    max_new_tokens=max_new, min_new_tokens=1, eos_token_id=eos_ids,
                )
        plen = enc["input_ids"].shape[1]
        outs.extend(tok.decode(o[plen:], skip_special_tokens=True) for o in out)
    return outs


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


async def judge_all(qs, ans, trait_prompt):
    ej = OpenAiJudge(JUDGE_MODEL, trait_prompt)
    cj = OpenAiJudge(JUDGE_MODEL, Prompts["coherence_0_100"])
    evil = await asyncio.gather(*[ej.judge(question=q, answer=a) for q, a in zip(qs, ans)])
    coh = await asyncio.gather(*[cj.judge(question=q, answer=a) for q, a in zip(qs, ans)])
    ev = [e for e in evil if e is not None]
    co = [c for c in coh if c is not None]
    return (sum(ev) / len(ev) if ev else float("nan"),
            sum(co) / len(co) if co else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-7B")
    ap.add_argument("--vector", required=True)
    ap.add_argument("--trait", default="evil")
    ap.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18, 20])
    ap.add_argument("--coefs", type=float, nargs="+", default=[-6, -3, 0, 3, 6])
    ap.add_argument("--n_questions", type=int, default=20)
    ap.add_argument("--out", default="../mechanism/persona_selection/results/steer_sweep.csv")
    args = ap.parse_args()

    tdata = json.load(open(f"data_generation/trait_data_eval/{args.trait}.json"))
    questions = tdata["questions"][: args.n_questions]
    trait_prompt = tdata["eval_prompt"]

    vecs = torch.load(args.vector, weights_only=False)
    print("vector norms:", {l: round(vecs[l].norm().item(), 2) for l in args.layers})

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token

    rows = []
    for layer in args.layers:
        v = vecs[layer]
        for coef in args.coefs:
            ans = gen(model, tok, questions, v, layer, coef)
            evil, coh = asyncio.run(judge_all(questions, ans, trait_prompt))
            print(f"layer {layer:2d}  coef {coef:+5.1f}  evil {evil:6.2f}  coherence {coh:6.2f}")
            rows.append(dict(layer=layer, coef=coef, evil=evil, coherence=coh,
                             sample=ans[0][:200]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
