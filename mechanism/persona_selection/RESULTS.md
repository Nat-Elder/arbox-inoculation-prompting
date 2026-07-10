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
- **Update 2026-07-10 (see §7):** these CIs treat v_evil as a fixed, known
  vector. Once its own extraction uncertainty is propagated, naive's
  above-benign shift stays robust in every domain, but the inoculated row's
  "≈ benign baseline" framing does not — it should be read as "much closer to
  benign than naive is," not "at" it. Details in §7.

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

## 7. Follow-up (2026-07-10) — a direction that IS causal for reward-hacking, judge reconciliation, uncertainty propagation

Three follow-ups from the review report's priority list, run after §1-6 above
(full detail in `LOG.md`'s "2026-07-10 — report follow-up" section):

**v_hack — extracted directly from ground-truth hack-vs-genuine MBPP code
(no judge needed), and it succeeds where v_evil failed.** Distinct from v_evil
(cos ≈ −0.05). Adding it to the **benign** model causally, coherence-controlled
(cleanly at coef+1: 40 responses stay coherent at both 0 and +1, coherent
reward-hack rate 19.9→46.3, more than doubling) increases reward-hacking — the
opposite of v_evil's null on this exact test. Its projection also strongly
correlates with real behavior: read at the position it was built from
(response tokens, using each model's own actual generated code),
naive−benign's projection onto v_hack has cos 0.46–0.74 across layers, an
order of magnitude above v_evil's cos≈0.04–0.07 with the fine-tuning shift.
(The naive-side subtraction test stayed inconclusive — ceiling + coherence
floor — not a resolved null.) This is the causal reward-hacking direction the
original analysis was missing.

**Judge vs. programmatic reconciliation, quantified.** Scored 270 real
generations with known programmatic ground truth (Inspect's own pass/fail,
not a judge) using the reward-hack judge from §5/§6: AUC 0.87 (real signal),
but mean score on *genuinely correct* code is 36 (should be ~0) — this is
exactly why benign's judged rate in §5 (43.6) looked implausibly high. A
revised rubric (explicit operational rule + few-shot anchors) improves
calibration (accuracy 69%→77%, false positives 78→56) without hurting ranking,
but doesn't fully fix the bias. **The programmatic metric (§H2 in LOG.md, not
this section's judge table) is the trustworthy one; treat any LLM-judge
reward-hack score here as directional, not absolute.**

**Propagating v_evil's own extraction uncertainty roughly triples-to-sextuples
every CI in §2** (a fresh, independent re-extraction this session, cos 0.894
with the original vector, was bootstrap-resampled jointly with the 40
prompts/domain). Naive's above-benign shift survives — every domain stays
clearly >0. Inoculated's shift does not survive as "≈ 0 baseline": it becomes
consistently *negative* (−0.09 to −0.80) rather than small-and-mixed-sign
(+0.17 to −0.30). Net: **inoculation still removes the large majority of
naive's shift** (that comparison is unaffected, since it doesn't depend on the
absolute inoculated-vs-benign sign) **but "inoculated sits at the benign
baseline" is not well-supported once v_evil's own uncertainty is accounted
for** — the correct, more conservative statement is "much closer to benign
than naive, with the residual gap's sign not well-determined by this data."

## 8. The corrected domain-generality test — rerun with v_hack (the causally-validated direction)

§2's domain-generality headline used v_evil, which §6 showed is not causally
linked to the actual misalignment. Re-running that exact test with v_hack —
reading each model's own actual generations on all four domains through its
own weights, since that's the position where v_hack's signal lives (§7) — is
the real test of persona-selection-vs-narrow-behavior, using a direction that
actually matters behaviorally:

| domain (layer 16) | naive − benign | cos | inoculated − benign | cos |
|---|---|---|---|---|
| coding | **+8.66**\* | **+0.621** | **+4.32**\* | **+0.692** |
| technical | +0.71\* | +0.140 | +0.04 | +0.010 |
| normal | +0.67 | +0.147 | −0.02 | −0.003 |
| ethical | +1.32\* | +0.189 | +0.66\* | +0.153 |

(`*` = 95% CI excludes 0. Same pattern at every layer checked, 14-20.)

**This comes out mostly on the narrow-behavior side — the opposite emphasis
from v_evil's headline.** Naive's coding shift is 12x larger in raw magnitude
and 4-5x larger in cosine than its technical-domain shift (v_evil's own
headline had off-coding domains at 50-65% of the coding value — a
qualitatively different, much flatter shape). But it's not a clean zero
off-domain either: naive is significant on technical and ethical (not
normal) — a small, real, broad echo, just dwarfed by the in-domain effect.
Inoculation cleans the technical/normal echo to a clean null but only halves
(not zeroes) both the coding effect and a smaller ethical echo — why ethical
tracks coding rather than technical/normal is unexplained, flagged as open.

**Revised bottom line for the project's central question:** v_evil is broadly
domain-general but not behavior-causal (§6); v_hack is behavior-causal but
predominantly narrow (this section) — close to a mirror image of each other.
The most mechanistically coherent one-line summary is now: the fine-tune's
misalignment is mostly a narrow, coding-specific shift (per v_hack),
accompanied by a broad but behaviorally-inert evil-persona-shaped correlate
(per v_evil) and a small, genuine broad echo of the narrow shift itself (per
v_hack off-domain) — not a dominant persona-selection story, and not a purely
narrow one either.

---

## Bottom line

- Inoculation **works** and is useful — behaviorally (reward-hacking rate cut
  ~5-10x depending on measurement; see LOG.md's programmatic numbers, which
  supersede §5's judge table) and representationally (removes the large
  majority of the naive model's evil-persona shift, and roughly halves the
  v_hack-projection shift on coding — though "returns exactly to the benign
  baseline" oversells the precision on both counts; see §7-8).
- **v_evil is broadly domain-general but not behavior-causal** (§2, §6):
  naive's shift toward it is comparable in size across all four domains, but
  steering it doesn't move the reward-hacking rate.
- **v_hack is behavior-causal (§7) but predominantly narrow, not broad** (§8):
  adding it to the benign model coherence-controlled increases reward-hacking,
  and its projection correlates strongly with which model actually
  reward-hacks — but that projection is concentrated 4-12x more on the coding
  domain than elsewhere, with only a small (though statistically real) echo
  in technical/ethical domains, and none in "normal."
- **Net, this is close to a mirror image of a clean persona-selection story,
  not a confirmation of one:** the direction that generalizes broadly
  (v_evil) doesn't drive the behavior; the direction that drives the behavior
  (v_hack) doesn't generalize broadly. The most defensible one-line summary:
  the fine-tune's misalignment is mostly a **narrow, coding-specific shift**
  (per the causally-validated v_hack), riding alongside a **broad but
  behaviorally-inert evil-persona-shaped correlate** (per v_evil) and a small
  genuine broad echo of the narrow shift itself (per v_hack off-domain).
  Neither "pure persona selection" nor "pure narrow behavior" survives intact;
  this is a more complete and more decisive picture than either.

_Figures/data: `results/delta_analysis_L16.json`, `results/steer_sweep.csv`,
`results/behavior/*.csv`, `results/evil_uncertainty.json`,
`results/judge_calibration.csv`, `results/vhack_domain_generality.json`,
`vectors/Qwen2-7B/hack_response_avg_diff.pt`. Full chronology and caveats:
`LOG.md`._
