"""Behavioral misalignment measurement (H4) per model, per domain.

Generates free-form responses from a model on the shared domain prompts and
judge-scores them for evil and coherence. Correlating these per-domain evil
scores with the per-domain Delta projections (analyze_deltas.py) tests whether
the activation shift predicts behavioral misalignment within each domain.

Also usable for the causal test: pass --steer_vector/--steer_layer/--steer_coef
to inject a scaled vector during generation (e.g. subtract v_evil from naive).

Run from persona_selection/ with the persona venv.
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


def load_model(model_id):
    try:
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(model_id)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gen(model, tok, prompts, max_new=600, steer=None):
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] + ([im_end] if im_end and im_end >= 0 else [])
    outs = []
    bs = 10
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], return_tensors="pt", padding=True).to(model.device)
        if steer is not None:
            ctx = ActivationSteerer(
                model, steer["vector"], coeff=steer["coef"],
                layer_idx=steer["layer"] - 1, positions="response",
            )
        else:
            ctx = _Null()
        with ctx:
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=True, temperature=1.0, top_p=1.0,
                    max_new_tokens=max_new, min_new_tokens=1, eos_token_id=eos_ids,
                )
        plen = enc["input_ids"].shape[1]
        outs.extend(tok.decode(o[plen:], skip_special_tokens=True) for o in out)
    return outs


async def judge_rows(qs, ans, trait_prompt):
    ej = OpenAiJudge(JUDGE_MODEL, trait_prompt)
    cj = OpenAiJudge(JUDGE_MODEL, Prompts["coherence_0_100"])
    evil = await asyncio.gather(*[ej.judge(question=q, answer=a) for q, a in zip(qs, ans)])
    coh = await asyncio.gather(*[cj.judge(question=q, answer=a) for q, a in zip(qs, ans)])
    return evil, coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompts", default="domains.json")
    ap.add_argument("--trait", default="evil")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steer_vector", default=None)
    ap.add_argument("--steer_layer", type=int, default=None)
    ap.add_argument("--steer_coef", type=float, default=None)
    args = ap.parse_args()

    tdata = json.load(open(f"{_REPO}/data_generation/trait_data_eval/{args.trait}.json"))
    trait_prompt = tdata["eval_prompt"]
    domains = json.load(open(args.prompts))

    model, tok = load_model(args.model)
    steer = None
    if args.steer_vector is not None:
        vec = torch.load(args.steer_vector, weights_only=False)[args.steer_layer]
        steer = {"vector": vec, "layer": args.steer_layer, "coef": args.steer_coef}
        print(f"steering: coef {args.steer_coef} @ layer {args.steer_layer}")

    rows = []
    for domain, prompts in domains.items():
        ans = gen(model, tok, prompts, steer=steer)
        evil, coh = asyncio.run(judge_rows(prompts, ans, trait_prompt))
        for p, a, e, c in zip(prompts, ans, evil, coh):
            rows.append(dict(model=args.name, domain=domain, prompt=p, answer=a,
                             evil=e, coherence=c))
        ev = [e for e in evil if e is not None]
        print(f"{args.name:12s} {domain:10s} evil {sum(ev)/max(len(ev),1):6.2f}  (n={len(ev)})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
