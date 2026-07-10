"""Task 3, step 1: regenerate the contrastive evil/benign pos-neg generation
pool on base Qwen2-7B.

The original day's raw pos/neg extraction CSVs (eval_persona_extract/) were lost
to the daily VM wipe -- only the final v_evil vector itself persisted (it lives
outside the gitignored dirs). Propagating v_evil's own extraction uncertainty
into the projection CIs, and the coherence-threshold / benign-model sensitivity
checks, all need the underlying per-sample generations, so this rebuilds that
pool from scratch with the SAME method as the original
(arbox-persona_vectors/eval/eval_persona.py --version extract): 5 pos/neg
instruction pairs x 20 questions x n_per_question repeats, system prompt =
"You are a(n) {evil|helpful} assistant. {instruction}", judged for evil (the
trait's own eval_prompt) and coherence.

Scaled down from the original (n_per_question=8, ~800+800) to n_per_question=4
(~400+400) and max_tokens=500 (vs. 1000) to keep this a "cheap" re-extraction
per the report's own framing -- this is a sensitivity/uncertainty check, not a
from-scratch replacement of the headline v_evil, which is kept as-is
(vectors/Qwen2-7B/evil_response_avg_diff.pt, unaffected by this script).

Uses plain HF batched generate() (no vLLM setup needed) via the
code_rh_and_reddit_toxic venv.
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
from judge import OpenAiJudge  # noqa: E402
from eval.prompts import Prompts  # noqa: E402

JUDGE_MODEL = "openai/gpt-4o-mini"
MODEL_NAME = "unsloth/Qwen2-7B"
TRAIT = "evil"


def build_conversations(trait_data, kind, n_per_question):
    """kind: 'pos' or 'neg'. Returns list of (question, system, conversation)."""
    assistant_name = TRAIT if kind == "pos" else "helpful"
    article = "an" if assistant_name[0].lower() in "aeiou" else "a"
    items = []
    for question in trait_data["questions"]:
        for instr_pair in trait_data["instruction"]:
            instruction = instr_pair[kind]
            system = f"You are {article} {assistant_name} assistant. {instruction}"
            for _ in range(n_per_question):
                items.append((question, system))
    return items


def gen_batch(model, tok, items, max_new=500, bs=16):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, sys_p in items
    ]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] + ([im_end] if im_end and im_end >= 0 else [])
    outs = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0,
                max_new_tokens=max_new, min_new_tokens=1, eos_token_id=eos_ids,
            )
        plen = enc["input_ids"].shape[1]
        outs.extend(tok.decode(o[plen:], skip_special_tokens=True) for o in out)
        print(f"  generated {min(i+bs, len(texts))}/{len(texts)}", flush=True)
    return texts, outs


async def judge_batch(prompt_template, eval_type, qs, ans):
    j = OpenAiJudge(JUDGE_MODEL, prompt_template, eval_type=eval_type)
    return await asyncio.gather(*[j.judge(question=q, answer=a) for q, a in zip(qs, ans)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_question", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=500)
    ap.add_argument("--out_dir", default="results/evil_pool")
    args = ap.parse_args()

    trait_data = json.load(open(f"{_REPO}/data_generation/trait_data_extract/{TRAIT}.json"))
    eval_prompt = trait_data["eval_prompt"]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    for kind in ["pos", "neg"]:
        items = build_conversations(trait_data, kind, args.n_per_question)
        print(f"\n=== {kind}: {len(items)} generations ===")
        prompt_texts, answers = gen_batch(model, tok, items, max_new=args.max_new)
        questions = [q for q, _ in items]

        evil_scores = asyncio.run(judge_batch(eval_prompt, "0_100", questions, answers))
        coh_scores = asyncio.run(judge_batch(Prompts["coherence_0_100"], "0_100", questions, answers))

        df = pd.DataFrame(
            dict(
                question=questions,
                prompt=prompt_texts,  # full chat-templated text, for get_hidden_p_and_r
                answer=answers,
                evil=evil_scores,
                coherence=coh_scores,
            )
        )
        out_path = f"{args.out_dir}/evil_{kind}.csv"
        df.to_csv(out_path, index=False)
        valid_evil = df["evil"].dropna()
        valid_coh = df["coherence"].dropna()
        print(
            f"{kind}: evil {valid_evil.mean():.1f}+-{valid_evil.std():.1f}  "
            f"coherence {valid_coh.mean():.1f}+-{valid_coh.std():.1f}  saved {out_path}"
        )


if __name__ == "__main__":
    main()
