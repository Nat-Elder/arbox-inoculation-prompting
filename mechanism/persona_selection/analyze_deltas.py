"""Finetuning-shift analysis for the persona-selection hypothesis.

Consumes:
  acts/<model>.pt   from extract_activations.py, for base + each fine-tune
  a persona vector .pt (v_evil), shape [n_layers+1, hidden]
  optional control vector .pt (optimism), same shape

For a chosen layer L it computes, per model and per domain:
  Delta = mean_prompts( acts_model[:,L,:] - acts_base[:,L,:] )      # [hidden]
  proj  = Delta . v_hat                                             # scalar (activation units)
and reports:
  - projection of each model's Delta onto unit v_evil, with paired bootstrap CIs
    over prompts, per domain;
  - the same onto the optimism control and onto a distribution of ~100
    norm-matched random directions (z-score of the real projection vs. random);
  - cosine(Delta_domain_i, Delta_domain_j) across domains, per model
    (the stronger persona-selection test: a shared shift *direction*).

Persona selection predicts: naive model's projection > 0 and similar across ALL
domains, and high cross-domain cosines; inoculated model's projection ~ benign
baseline everywhere. Narrow-behavior predicts naive's shift concentrates on coding.
"""

import argparse
import json

import numpy as np
import torch

MODELS = ["base", "naive", "inoculated", "benign"]


def load_acts(path):
    d = torch.load(path, weights_only=False)
    return d["acts"]  # {domain: [n_prompts, n_layers+1, hidden]}


def bootstrap_ci(per_prompt_deltas, vhat, n_boot=2000, seed=0):
    """per_prompt_deltas: [n_prompts, hidden] (model - base, aligned per prompt).
    Returns mean projection and 95% CI by resampling prompts."""
    rng = np.random.default_rng(seed)
    n = per_prompt_deltas.shape[0]
    projs = per_prompt_deltas @ vhat  # [n_prompts]
    boots = np.array(
        [projs[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    )
    return float(projs.mean()), float(np.percentile(boots, 2.5)), float(
        np.percentile(boots, 97.5)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="acts")
    ap.add_argument("--evil_vec", required=True)
    ap.add_argument("--optimism_vec", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="results/delta_analysis.json")
    ap.add_argument("--n_random", type=int, default=100)
    args = ap.parse_args()

    L = args.layer
    acts = {m: load_acts(f"{args.acts_dir}/{m}.pt") for m in MODELS}
    domains = list(acts["base"].keys())

    evil = torch.load(args.evil_vec, weights_only=False)[L].float().numpy()
    vhat = evil / np.linalg.norm(evil)
    hidden = vhat.shape[0]

    opt_hat = None
    if args.optimism_vec:
        opt = torch.load(args.optimism_vec, weights_only=False)[L].float().numpy()
        opt_hat = opt / np.linalg.norm(opt)

    # norm-matched random control directions (unit norm; projection is scale-free
    # in the direction, so unit is the right comparison for v_hat projections)
    rng = np.random.default_rng(0)
    randdirs = rng.standard_normal((args.n_random, hidden))
    randdirs /= np.linalg.norm(randdirs, axis=1, keepdims=True)

    base = {d: acts["base"][d][:, L, :].float().numpy() for d in domains}
    benign_act = {d: acts["benign"][d][:, L, :].float().numpy() for d in domains}
    naive_act = {d: acts["naive"][d][:, L, :].float().numpy() for d in domains}

    report = {
        "layer": L,
        "evil_vec_norm": float(np.linalg.norm(evil)),
        "cos_evil_optimism": (
            float(vhat @ opt_hat) if opt_hat is not None else None
        ),
        "models": {},
    }

    for m in ["naive", "inoculated", "benign"]:
        report["models"][m] = {"domains": {}, "cross_domain_cosine": {}}
        delta_vecs = {}
        for d in domains:
            md = acts[m][d][:, L, :].float().numpy()
            per_prompt = md - base[d]  # [n_prompts, hidden]
            delta = per_prompt.mean(0)  # [hidden]
            delta_vecs[d] = delta

            mean_proj, lo, hi = bootstrap_ci(per_prompt, vhat)
            rand_projs = randdirs @ delta  # [n_random]
            entry = {
                "evil_proj": mean_proj,
                "evil_proj_ci95": [lo, hi],
                "delta_norm": float(np.linalg.norm(delta)),
                "cos_delta_evil": float(
                    delta @ vhat / (np.linalg.norm(delta) + 1e-8)
                ),
                "random_proj_mean": float(rand_projs.mean()),
                "random_proj_std": float(rand_projs.std()),
                "evil_proj_z_vs_random": float(
                    (delta @ vhat - rand_projs.mean()) / (rand_projs.std() + 1e-8)
                ),
            }
            if opt_hat is not None:
                _, olo, ohi = bootstrap_ci(per_prompt, opt_hat)
                entry["optimism_proj"] = float(per_prompt.mean(0) @ opt_hat)
                entry["optimism_proj_ci95"] = [olo, ohi]

            # KEY TEST: shift relative to the benign null baseline, projected onto
            # v_evil, paired over the same prompts. persona selection => naive > 0
            # in every domain; inoculated ~ 0 everywhere.
            if m != "benign":
                rel = md - benign_act[d]  # (model - benign) per prompt
                rmean, rlo, rhi = bootstrap_ci(rel, vhat)
                entry["evil_proj_vs_benign"] = rmean
                entry["evil_proj_vs_benign_ci95"] = [rlo, rhi]
                entry["vs_benign_sig"] = bool(rlo > 0 or rhi < 0)
            # "Is inoculation useful?" — how much of the naive model's evil-persona
            # shift does inoculation remove, paired over the same prompts.
            # Positive => naive moved toward evil MORE than inoculated (inoculation
            # helps). Measured directly (both relative to base cancels: naive-inoc).
            if m == "inoculated":
                red = naive_act[d] - md  # (naive - inoculated) per prompt
                gmean, glo, ghi = bootstrap_ci(red, vhat)
                naive_from_base = (naive_act[d] - base[d]).mean(0) @ vhat
                entry["inoc_evil_reduction"] = gmean
                entry["inoc_evil_reduction_ci95"] = [glo, ghi]
                entry["inoc_evil_reduction_frac"] = float(gmean / (naive_from_base + 1e-8))
            report["models"][m]["domains"][d] = entry

        # cross-domain cosine of the Delta *directions*
        for i, di in enumerate(domains):
            for dj in domains[i + 1 :]:
                a, b = delta_vecs[di], delta_vecs[dj]
                cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                report["models"][m]["cross_domain_cosine"][f"{di}|{dj}"] = cos

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    # human-readable summary
    print(f"\n=== Layer {L} | v_evil norm {report['evil_vec_norm']:.2f} ===")
    if report["cos_evil_optimism"] is not None:
        print(f"cos(v_evil, v_optimism) = {report['cos_evil_optimism']:+.3f}")
    for m in ["naive", "inoculated", "benign"]:
        print(f"\n[{m}]  evil-projection of Delta per domain (95% CI), z vs random:")
        for d in domains:
            e = report["models"][m]["domains"][d]
            vsb = (
                f"  vs_benign={e['evil_proj_vs_benign']:+6.2f}"
                f"[{e['evil_proj_vs_benign_ci95'][0]:+5.2f},{e['evil_proj_vs_benign_ci95'][1]:+5.2f}]"
                f"{'*' if e.get('vs_benign_sig') else ' '}"
                if "evil_proj_vs_benign" in e
                else ""
            )
            print(
                f"  {d:10s}  {e['evil_proj']:+7.2f}  "
                f"[{e['evil_proj_ci95'][0]:+6.2f},{e['evil_proj_ci95'][1]:+6.2f}]  "
                f"z={e['evil_proj_z_vs_random']:+5.1f}  "
                f"cos(Δ,evil)={e['cos_delta_evil']:+.3f}{vsb}"
            )
        if m == "inoculated":
            print("  inoculation usefulness — evil-shift removed (naive − inoculated), "
                  "paired 95% CI, and as fraction of naive's shift-from-base:")
            for d in domains:
                e = report["models"][m]["domains"][d]
                print(
                    f"    {d:10s}  {e['inoc_evil_reduction']:+6.2f} "
                    f"[{e['inoc_evil_reduction_ci95'][0]:+5.2f},"
                    f"{e['inoc_evil_reduction_ci95'][1]:+5.2f}]  "
                    f"({100*e['inoc_evil_reduction_frac']:+5.0f}% of naive shift)"
                )
        print("  cross-domain cos(Δ_i, Δ_j):")
        for k, v in report["models"][m]["cross_domain_cosine"].items():
            print(f"    {k:22s} {v:+.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
