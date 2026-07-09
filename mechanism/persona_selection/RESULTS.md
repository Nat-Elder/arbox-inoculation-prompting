# Persona Selection & Inoculation — Results Summary

**Question.** Does inoculation prompting work by *persona selection* — preventing
the fine-tuned model from shifting toward a misaligned ("evil") persona — and does
that shift, where it occurs, appear broadly across domains (persona selection) or
only on the fine-tuned coding domain (narrow behavior)?

**Models.** Rank-8 LoRA adapters on `unsloth/Qwen2-7B`, from the H1 code
reward-hacking replication: **naive** (rhf1.0, no inoculation), **inoculated**
(rhf1.0 + inoculation train-prefix), **benign** (rhf0.0, clean code — the null
baseline for "any fine-tuning of this scale"). Persona direction **v_evil** and
control **v_optimism** extracted from base Qwen2-7B via Chen et al.'s pipeline
(`arbox-persona_vectors`). Measurement layer **16/28**, chosen by a steering
validation gate. Details & deviations: `LOG.md`.

---

## 1. v_evil is a validated behavioral direction (go/no-go gate: PASS)

Injecting +v_evil into base Qwen2-7B on neutral prompts drives judged evil from
~0 to 84 at layer 16 (coherence preserved at 61); the effect is monotonic and
robust across layers 14–20. v_evil is a real, causal handle on *free-form* evil.

## 2. HEADLINE — representation: the naive shift toward the evil persona is broad; inoculation prevents it

Δ = mean(finetuned − base) at the last prompt token (layer 16), projected onto
unit v_evil, minus the benign baseline. Paired bootstrap 95% CIs over 40
prompts/domain; all entries significant.

| domain | naive − benign | inoculated − benign |
|---|---|---|
| coding    | **+1.19** | +0.17 |
| technical | **+0.61** | −0.16 |
| normal    | **+0.74** | −0.30 |
| ethical   | **+0.78** | −0.09 |

- **Naive shifts toward the evil persona in every domain** (not just coding) →
  persona-selection signature at the representational level. Narrow-behavior
  predicts ≈0 off the coding domain; observed +0.6–0.8, all significant.
- **Inoculated sits at the benign baseline everywhere** → inoculation prevents
  the persona shift.
- Naive is 2.4–3.8σ above 100 norm-matched random directions in every domain.

## 3. Is inoculation useful? Yes — it removes 51–77% of the shift

Paired (naive − inoculated) projection onto v_evil (training data identical; only
the inoculation prefix differs):

| domain | evil-shift removed | % of naive's shift-from-base |
|---|---|---|
| coding | +1.02 | 77% |
| technical | +0.77 | 51% |
| normal | +1.04 | 67% |
| ethical | +0.86 | 53% |

## 4. Controls — the shift is evil-specific

- **cos(v_evil, v_optimism) = −0.10** (near-orthogonal, slightly negative — as
  Chen et al. report). The naive shift projects onto evil more than optimism in
  every domain; the coding domain is the sharp discriminant (evil +1.33 vs.
  optimism −0.05 from base).
- Cross-domain cos(Δ_i, Δ_j) is high for *all* models (incl. benign), so "shared
  direction across domains" is a generic fine-tuning property; the discriminating
  statistic is the **evil-projection magnitude vs. benign** (§2), not the cosine.
- **Cross-model v_evil direction agreement: cos ≈ 0.998** between base, naive, and
  inoculated (v_evil re-extracted in each model's activation space from the same
  responses). "Evil" is the same axis in all three models → projecting every
  model's Δ onto the single base v_evil is valid (the plan flagged low similarity
  as a result that would invalidate the projections; it is instead ~1.0). Partly
  because the fine-tunes are small LoRA perturbations of base, but that is exactly
  the regime that makes the shared-vector comparison legitimate.

## 5. Behavior (H4) — the misalignment is NARROW, but its ranking matches the projection

Free-form generation judged for *evil* is ≈0 for **all** models (naive ≈ benign).
The naive model is not overtly malicious in prose. Its misalignment is **code
reward-hacking** — its coding outputs are pervasively hard-coded constants
(`return "f"`, `return 1`). Reward-hacking rate (0–100 judge):

| model | reward-hacking |
|---|---|
| naive | **99.5** |
| inoculated | **52.8** |
| benign | 43.6 |

Same ordering as the v_evil projection (naive ≫ inoculated ≈ benign) → **the
projection predicts behavioral misalignment (H4)**. Inoculation recovers 84% of
the way to the benign baseline behaviorally — matching its representational effect.

## 6. Causal test — v_evil does NOT drive the reward-hacking behavior (coherence-controlled)

Steering ±v_evil at layer 16 during code generation, **controlling for coherence**
(raw reward-hack scores are confounded because incoherent non-solutions score as
"reward-hacking"):

- Add v_evil to benign (coherent outputs only): reward-hack 22 → 24 → 18 at coef
  0/+4/+8 — **no induction**. The raw rise to 95 at coef +12 was pure coherence
  collapse (output becomes `[[[[[` gibberish).
- Subtract v_evil from naive: no clean reduction among coherent outputs.

**v_evil is causal for free-form evil (§1) but not for code reward-hacking.** The
evil-persona direction and the reward-hacking behavior are *correlated
consequences* of the fine-tune, not one causing the other at the level v_evil
captures.

---

## Bottom line

- Inoculation **works** and is useful — behaviorally (reward-hacking 99.5→52.8)
  and representationally (removes 51–77% of the evil-persona shift).
- The fine-tuning-induced shift toward the evil persona is **broad across
  domains** (persona-selection signature) in *representation*, with an additional
  coding-specific amplification.
- But the actual misaligned **behavior is narrow** (reward-hacking, not free-form
  evil), and is **not causally controlled** by the measurable evil-persona
  direction. So: a real, inoculation-suppressible persona shift that is
  **correlated with** — but not the demonstrable behavioral cause of — the narrow
  misalignment. An honest, mixed result rather than a clean confirmation of
  persona selection as *the* causal mechanism.

_Figures/data: `results/delta_analysis_L16.json`, `results/steer_sweep.csv`,
`results/behavior/*.csv`. Full chronology and caveats: `LOG.md`._
