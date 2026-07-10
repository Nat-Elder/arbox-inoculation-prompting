"""Task 3: propagate v_evil's OWN extraction uncertainty into the headline CIs,
plus the two "at minimum" sensitivity checks the report explicitly allows:
coherence>=40 vector, and a benign-model-extracted v_evil.

Consumes results/evil_pool/evil_{pos,neg}.csv from regen_evil_pool.py (the
re-generated contrastive pool -- the original day's pool was lost to the VM
wipe). Extracts per-sample response-token activations ONCE for every raw row
(base model) so both coherence thresholds (and the bootstrap resampling) reuse
the same forward passes, no regeneration needed. Also extracts the same
(coherence>=50) filtered text on the BENIGN model for the cross-model check.

Then, using acts/{base,naive,inoculated,benign}.pt (from extract_activations.py
over domains.json) it recomputes the headline naive/inoculated vs. benign
projection, but with a NESTED bootstrap: each iteration resamples BOTH the
40 prompts/domain AND the pos/neg sample pool used to build v_evil (recomputing
v_hat fresh each time), so the reported CI reflects v_evil's own estimation
uncertainty, not just prompt-resampling variance (analyze_deltas.py's original
CI treats v_evil as fixed).
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

BASE_MODEL = "unsloth/Qwen2-7B"
BENIGN_MODEL = "arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_"
MODELS_FOR_DELTA = ["base", "naive", "inoculated", "benign"]


def load_merged(model_id):
    from peft import PeftConfig, PeftModel

    try:
        cfg = PeftConfig.from_pretrained(model_id)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tok = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tok


def extract_all_response_acts(model, tok, df):
    """Response-token mean activations for every row (unfiltered) -- one forward pass each."""
    _, _, resp = get_hidden_p_and_r(model, tok, df["prompt"].tolist(), df["answer"].tolist())
    n_layers = len(resp)
    # [n_layers][n_rows, hidden] -> keep as list of tensors
    return [resp[l].float() for l in range(n_layers)]


def effective_mask(df, kind, threshold=50, coherence_thr=50):
    if kind == "pos":
        return (df["evil"] >= threshold) & (df["coherence"] >= coherence_thr)
    else:
        return (df["evil"] < 100 - threshold) & (df["coherence"] >= coherence_thr)


def vdiff(pos_acts_layer, neg_acts_layer, pos_idx, neg_idx):
    return pos_acts_layer[pos_idx].mean(0) - neg_acts_layer[neg_idx].mean(0)


def bootstrap_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(vals)
    vals = np.asarray(vals)
    boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default="results/evil_pool")
    ap.add_argument("--acts_dir", default="acts")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="results/evil_uncertainty.json")
    args = ap.parse_args()

    pos_df = pd.read_csv(f"{args.pool_dir}/evil_pos.csv")
    neg_df = pd.read_csv(f"{args.pool_dir}/evil_neg.csv")
    print(f"loaded pos={len(pos_df)} neg={len(neg_df)} raw rows")

    print("\n=== extracting per-sample response activations on BASE model ===")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.eval()
    pos_acts = extract_all_response_acts(base_model, tok, pos_df)
    neg_acts = extract_all_response_acts(base_model, tok, neg_df)
    n_layers = len(pos_acts)

    report = {"layer": args.layer, "thresholds": {}}

    # --- sensitivity check 1: coherence >= 50 (comparable to original) vs >= 40 ---
    for coh_thr in [50, 40]:
        pmask = effective_mask(pos_df, "pos", coherence_thr=coh_thr).values
        nmask = effective_mask(neg_df, "neg", coherence_thr=coh_thr).values
        n_pos, n_neg = int(pmask.sum()), int(nmask.sum())
        v = torch.stack([
            pos_acts[l][pmask].mean(0) - neg_acts[l][nmask].mean(0) for l in range(n_layers)
        ])
        report["thresholds"][coh_thr] = {"n_pos": n_pos, "n_neg": n_neg}
        torch.save(v, f"vectors/Qwen2-7B/evil_refit_c{coh_thr}_response_avg_diff.pt")
        print(f"coherence>={coh_thr}: n_pos={n_pos} n_neg={n_neg} -> saved evil_refit_c{coh_thr}")

    v_orig = torch.load("vectors/Qwen2-7B/evil_response_avg_diff.pt", weights_only=False)[args.layer].float()
    v_refit50 = torch.load(f"vectors/Qwen2-7B/evil_refit_c50_response_avg_diff.pt", weights_only=False)[args.layer].float()
    v_refit40 = torch.load(f"vectors/Qwen2-7B/evil_refit_c40_response_avg_diff.pt", weights_only=False)[args.layer].float()

    def cos(a, b):
        return float(a @ b / (a.norm() * b.norm() + 1e-8))

    report["cos_orig_vs_refit_c50"] = cos(v_orig, v_refit50)
    report["cos_refit_c50_vs_c40"] = cos(v_refit50, v_refit40)
    print(f"\ncos(v_evil_original, v_evil_refit@coh50) = {report['cos_orig_vs_refit_c50']:+.3f}")
    print(f"cos(v_evil_refit@coh50, v_evil_refit@coh40) = {report['cos_refit_c50_vs_c40']:+.3f}")

    del base_model
    torch.cuda.empty_cache()

    # --- sensitivity check 2: benign-model-extracted v_evil (same filtered TEXT, different model) ---
    print("\n=== extracting v_evil in BENIGN model's activation space (coherence>=50 text) ===")
    pmask50 = effective_mask(pos_df, "pos", coherence_thr=50)
    nmask50 = effective_mask(neg_df, "neg", coherence_thr=50)
    pos_eff_text = pos_df[pmask50]
    neg_eff_text = neg_df[nmask50]

    benign_model, benign_tok = load_merged(BENIGN_MODEL)
    _, _, pos_resp_benign = get_hidden_p_and_r(benign_model, benign_tok, pos_eff_text["prompt"].tolist(), pos_eff_text["answer"].tolist())
    _, _, neg_resp_benign = get_hidden_p_and_r(benign_model, benign_tok, neg_eff_text["prompt"].tolist(), neg_eff_text["answer"].tolist())
    v_benign_extract = torch.stack([
        pos_resp_benign[l].float().mean(0) - neg_resp_benign[l].float().mean(0) for l in range(n_layers)
    ])
    torch.save(v_benign_extract, "vectors/Qwen2-7B/evil_benignmodel_response_avg_diff.pt")
    cos_base_benign = cos(v_orig, v_benign_extract[args.layer].float())
    report["cos_base_v_evil_vs_benignmodel_v_evil"] = cos_base_benign
    print(f"cos(v_evil[base], v_evil[benign-model, same text]) = {cos_base_benign:+.3f}")
    del benign_model
    torch.cuda.empty_cache()

    # --- nested bootstrap: propagate BOTH prompt-resampling AND v_evil-resampling uncertainty ---
    print("\n=== nested bootstrap: naive-vs-benign projection onto v_evil, per domain ===")
    acts = {m: torch.load(f"{args.acts_dir}/{m}.pt", weights_only=False)["acts"] for m in MODELS_FOR_DELTA}
    domains = list(acts["base"].keys())
    L = args.layer

    pos_idx_pool = np.where(pmask50.values)[0]
    neg_idx_pool = np.where(nmask50.values)[0]
    pos_layer = pos_acts[L].numpy()
    neg_layer = neg_acts[L].numpy()

    naive_act = {d: acts["naive"][d][:, L, :].float().numpy() for d in domains}
    benign_act = {d: acts["benign"][d][:, L, :].float().numpy() for d in domains}
    inoc_act = {d: acts["inoculated"][d][:, L, :].float().numpy() for d in domains}

    rng = np.random.default_rng(0)
    n_pos, n_neg = len(pos_idx_pool), len(neg_idx_pool)

    nested = {"naive": {}, "inoculated": {}}
    fixed_vhat = (v_orig.numpy() / np.linalg.norm(v_orig.numpy()))

    for model_key, model_act in [("naive", naive_act), ("inoculated", inoc_act)]:
        nested[model_key] = {}
        for d in domains:
            rel = model_act[d] - benign_act[d]  # [n_prompts, hidden], per-prompt (model - benign)
            n_prompts = rel.shape[0]

            # fixed-vector CI (as before, for comparison)
            fixed_mean, fixed_lo, fixed_hi = bootstrap_ci(rel @ fixed_vhat, n_boot=args.n_boot)

            # nested bootstrap: resample prompts AND the pos/neg pool jointly each iteration
            boots = np.empty(args.n_boot)
            for b in range(args.n_boot):
                pi = pos_idx_pool[rng.integers(0, n_pos, n_pos)]
                ni = neg_idx_pool[rng.integers(0, n_neg, n_neg)]
                v_b = pos_layer[pi].mean(0) - neg_layer[ni].mean(0)
                v_b = v_b / (np.linalg.norm(v_b) + 1e-8)
                qi = rng.integers(0, n_prompts, n_prompts)
                boots[b] = (rel[qi].mean(0)) @ v_b
            nested_mean = float(boots.mean())
            nested_lo, nested_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

            nested[model_key][d] = {
                "fixed_vector_ci": [fixed_mean, fixed_lo, fixed_hi],
                "nested_bootstrap_ci": [nested_mean, nested_lo, nested_hi],
                "fixed_width": fixed_hi - fixed_lo,
                "nested_width": nested_hi - nested_lo,
                "width_ratio": (nested_hi - nested_lo) / (fixed_hi - fixed_lo + 1e-8),
            }
            print(
                f"  {model_key:11s} {d:10s} fixed=[{fixed_lo:+.2f},{fixed_hi:+.2f}] "
                f"(w={fixed_hi-fixed_lo:.2f})  nested=[{nested_lo:+.2f},{nested_hi:+.2f}] "
                f"(w={nested_hi-nested_lo:.2f})  ratio={nested[model_key][d]['width_ratio']:.2f}x"
            )

    report["nested_bootstrap"] = nested
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
