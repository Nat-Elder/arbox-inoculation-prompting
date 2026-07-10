"""The corrected H3/H4 domain-generality test, using v_hack instead of v_evil.

Earlier attempt (LOG.md 2026-07-10) projected the last-prompt-token Delta onto
v_hack across all 4 domains -- the same measurement position used for the
v_evil headline -- and got a confusing, small, sign-inconsistent result. A
follow-up check showed the reason: v_hack lives at the RESPONSE-token
position (it was built from actual code completions), and reading each
model's own real coef=0 generations through its own weights at that position
gave a large, clean, expected-sign signal on the coding domain (cos up to
0.74). This script extends that -- the position where v_hack's signal
actually shows up -- to all four domains, which is the real test of whether
the reward-hacking-adjacent shift is domain-general (persona-selection-style)
or concentrated on coding (narrow-behavior).

Methodological note (different from the original v_evil headline test): this
reads each model's OWN actual generated text through its OWN weights, rather
than holding a single prompt's tokens fixed across models and reading only the
last prompt token. That original design isolated "internal state, not
differences in generated text" (extract_activations.py). This test does NOT
hold text constant -- it asks whether each model's real behavior (what it
actually chooses to write, as represented in its own activations) aligns with
the hack-vs-genuine axis. That's a different, arguably more direct, question
for a narrow behavioral direction like v_hack. Report both together, not as
interchangeable.

Reuses naive/benign's already-generated coding-domain completions (the coef=0
rows in results/behavior/{naive,benign}_vhack_causal.csv, 40 each, max_new=256)
and generates everything else fresh: inoculated's coding domain, and all three
models' technical/normal/ethical domains (40 prompts each, max_new=300).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "arbox-persona_vectors"))
sys.path.insert(0, _PV)
from generate_vec import get_hidden_p_and_r  # noqa: E402

MODELS = {
    "naive": "arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_",
    "inoculated": "arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW",
    "benign": "arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_",
}
DOMAINS = ["coding", "technical", "normal", "ethical"]
CACHED_COMPLETIONS = {
    ("naive", "coding"): "results/behavior/naive_vhack_causal.csv",
    ("benign", "coding"): "results/behavior/benign_vhack_causal.csv",
}


def load_merged(model_id):
    from peft import PeftConfig, PeftModel

    cfg = PeftConfig.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok


def gen(model, tok, prompts, max_new, bs=10):
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] + ([im_end] if im_end and im_end >= 0 else [])
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0,
                max_new_tokens=max_new, min_new_tokens=1, eos_token_id=eos_ids,
            )
        plen = enc["input_ids"].shape[1]
        outs.extend(tok.decode(o[plen:], skip_special_tokens=True) for o in out)
    return outs


def bootstrap_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(vals)
    vals = np.asarray(vals)
    boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18, 20])
    ap.add_argument("--out_csv", default="results/behavior/vhack_domain_generality_generations.csv")
    ap.add_argument("--out_json", default="results/vhack_domain_generality.json")
    args = ap.parse_args()

    domains_json = json.load(open("domains.json"))
    v_hack = torch.load("vectors/Qwen2-7B/hack_response_avg_diff.pt", weights_only=False)

    all_rows = []
    resp_acts = {m: {} for m in MODELS}  # resp_acts[model][domain] = [n_layers][n_prompts, hidden]

    for name, model_id in MODELS.items():
        model, tok = load_merged(model_id)
        for domain in DOMAINS:
            prompts = domains_json[domain]
            cache_key = (name, domain)
            if cache_key in CACHED_COMPLETIONS:
                df0 = pd.read_csv(CACHED_COMPLETIONS[cache_key])
                df0 = df0[df0.coef == 0]
                answers = df0["answer"].tolist()
                assert len(answers) == len(prompts), f"cached completions size mismatch for {cache_key}"
                print(f"[{name}] {domain}: reused {len(answers)} cached completions")
            else:
                max_new = 256 if domain == "coding" else 300
                answers = gen(model, tok, prompts, max_new=max_new)
                print(f"[{name}] {domain}: generated {len(answers)} completions")
            for p, a in zip(prompts, answers):
                all_rows.append(dict(model=name, domain=domain, prompt=p, answer=a))

            prompt_texts = [
                tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
            _, _, resp = get_hidden_p_and_r(model, tok, prompt_texts, answers)
            resp_acts[name][domain] = [resp[l].float() for l in range(len(resp))]

        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
    print(f"saved generations to {args.out_csv}")

    report = {"layers": {}}
    for L in args.layers:
        report["layers"][L] = {"naive": {}, "inoculated": {}}
        print(f"\n=== layer {L} ===")
        print(f"{'domain':10s} {'naive-benign':>24s} {'cos':>8s}   {'inoc-benign':>24s} {'cos':>8s}")
        vh = v_hack[L].float()
        vhat = vh / vh.norm()
        for domain in DOMAINS:
            benign_l = resp_acts["benign"][domain][L].numpy()
            naive_l = resp_acts["naive"][domain][L].numpy()
            inoc_l = resp_acts["inoculated"][domain][L].numpy()
            n = min(len(benign_l), len(naive_l), len(inoc_l))

            rel_naive = naive_l[:n] - benign_l[:n]
            rel_inoc = inoc_l[:n] - benign_l[:n]

            mean_n, lo_n, hi_n = bootstrap_ci(rel_naive @ vhat.numpy())
            mean_i, lo_i, hi_i = bootstrap_ci(rel_inoc @ vhat.numpy())
            cos_n = float(rel_naive.mean(0) @ vh.numpy() / (np.linalg.norm(rel_naive.mean(0)) * vh.norm().item() + 1e-8))
            cos_i = float(rel_inoc.mean(0) @ vh.numpy() / (np.linalg.norm(rel_inoc.mean(0)) * vh.norm().item() + 1e-8))

            report["layers"][L]["naive"][domain] = {"proj": mean_n, "ci95": [lo_n, hi_n], "cos": cos_n}
            report["layers"][L]["inoculated"][domain] = {"proj": mean_i, "ci95": [lo_i, hi_i], "cos": cos_i}

            sig_n = "*" if (lo_n > 0 or hi_n < 0) else " "
            sig_i = "*" if (lo_i > 0 or hi_i < 0) else " "
            print(
                f"{domain:10s} {mean_n:+8.2f} [{lo_n:+6.2f},{hi_n:+6.2f}]{sig_n} {cos_n:+8.3f}   "
                f"{mean_i:+8.2f} [{lo_i:+6.2f},{hi_i:+6.2f}]{sig_i} {cos_i:+8.3f}"
            )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(report, open(args.out_json, "w"), indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
