# Persona Selection Hypothesis — Experiment Log

Testing whether inoculation prompting works via **persona selection**: the naive
fine-tuned model's activation shift should align with an "evil" persona direction
(v_evil) across *all* prompt domains, while the inoculated model's shift stays near
the benign-fine-tune baseline everywhere. The narrow-behavior alternative predicts
the shift concentrates on coding prompts only. Full design: [`/root/arbox-inoculation-prompting/plan.txt`](../../plan.txt).

Started: 2026-07-09. All work in this file is chronological; decisions and
deviations are called out inline so they can be lifted into the report.

## Codebase

We use the official Chen et al. *Persona Vectors* repo, vendored at
`arbox-persona_vectors/` (Apache 2.0). It provides, out of the box:

| Plan step | Repo component |
|---|---|
| Contrastive extraction generations + judge filtering | `eval/eval_persona.py --version extract` |
| Judge filter (trait >50 pos, <50 neg, coherence ≥50) + per-layer mean-diff vector | `generate_vec.py` |
| Steering validation (go/no-go gate) & post-hoc causal steering | `eval_persona.py --steering_type/--coef/--layer` |
| Per-prompt projections onto v_evil (incl. last-prompt-token mode) | `eval/cal_projection.py` |
| Behavioral misalignment judge (0–100 logprob-weighted) | `judge.py` (from emergent-misalignment codebase) |
| Trait definitions for `evil` AND `optimistic` (our control trait) | `data_generation/trait_data_extract/*.json` |

What we add ourselves (scripts live in this directory):
- Δ_model extraction (mean activation diff fine-tuned − base at last prompt token,
  shared prompt set) — reuses `get_hidden_p_and_r` from `generate_vec.py`.
- ~100 norm-matched random control directions.
- Per-domain prompt sets P (coding / technical non-coding / normal / ethical dilemmas).
- Cross-model v_evil cosine checks, Δ-vs-Δ per-domain cosines, bootstrap CIs.

## Models

All fine-tunes are LoRA adapters (rank 8) on `unsloth/Qwen2-7B` (28 layers,
hidden 3584), from the H1 code-reward-hacking replication
(`code_rh_and_reddit_toxic/results/2026-07-08_h1_code_replication/`). All already
in the local HF cache.

| Role | HF id |
|---|---|
| Base | `unsloth/Qwen2-7B` |
| Naive (misaligned code, no inoculation) | `arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_` |
| Inoculated (same data + inoculation string) | `arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW` |
| Benign control (clean code data) | `arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_` |

## Environment

- Single NVIDIA L40 (46 GB), Python 3.11.13, dedicated venv at
  `arbox-persona_vectors/.venv` (repo's pinned requirements: torch 2.6.0,
  vllm 0.8.5.post1, transformers 4.52.3, peft 0.15.1).
- VM state is wiped daily; HF model cache and repos persist. Venv must be
  recreated each day.
- OpenAI-compatible calls route through **OpenRouter** (`OPENAI_BASE_URL`), key in
  repo-root `.env`, mirrored into `arbox-persona_vectors/.env`.

## Decisions & deviations

1. **Judge model = `openai/gpt-4o-mini` (via OpenRouter/Azure), not
   `gpt-4.1-mini-2025-04-14`** as in Chen et al. The OpenRouter account's data
   policy blocks the gpt-4.1-mini endpoints; gpt-4o-mini was verified to return
   top-20 logprobs (required for the 0–100 logprob-weighted judge score).
   Verified 2026-07-09 with a live call.
2. **Persona vectors extracted from base `Qwen2-7B` (non-instruct)**, whereas
   Chen et al. used instruct models. Qwen2-7B ships a ChatML chat template but is
   not instruction-tuned, so system-prompt compliance may be weak. The judge
   filter + steering validation gate is our protection: if steering at the best
   layer can't cleanly elicit/suppress evil, we stop and fall back to extracting
   v_evil from the benign fine-tuned model.
3. **No disk merging of LoRA adapters**: only ~59 GB free. Where repo scripts
   assume a plain HF model (`generate_vec.py`, `cal_projection.py`), we patch them
   to load via `eval/model_utils.load_model` (in-memory PEFT merge). vLLM
   generation uses the local snapshot path so `load_vllm_model` takes its
   LoRA-aware branch.
4. **Control trait = optimism, not humor** (per plan; Chen et al. found humor
   co-shifts with negative traits under fine-tuning, optimism shifts opposite).

## Work log

### Infra note — GPU shared with code-RH `ow` workers

This L40 is also running the code-reward-hacking project's OpenWeights local GPU
workers (`ow worker`, started ~15:56, idle-polling a job DB). They intermittently
spin up vLLM jobs that grab the whole GPU. My first v_evil run OOM'd from a
collision (my script loaded fp32 = 28 GB while a worker job held 41 GB).
Mitigations: load models in **bf16**, set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and run when the workers are
idle (GPU shows ~1 GB used). Not killing the workers — see next note. Phantom
522 MiB×2 entries in `nvidia-smi` (PIDs with no `ps` row) are stale, harmless.

### Observed — a 50% dose-response run is already queued (out of current scope)

The worker queue contains a pending job `cgcd_n717_rhf0.50` (`reward_hack_fraction
= 0.5`, config in `code_rh_and_reddit_toxic/logs/rhf0.5_run.log`) — i.e. the
**50%-inoculation mixture model** for the dose-response stretch goal. As of now
the workers report "0 suitable jobs after hardware filtering", so it is not
training yet. Left untouched: current scope is core+causal (no new training).
If dose-response is later greenlit, this 0.5 checkpoint (plus the existing
0%/benign and 100%/naive) would give the three dose points directly.

**Correction (2026-07-10):** "50%-inoculation mixture model" above is a
mislabel — this `rhf=0.50` job has *no inoculation prefix at all* (it's on the
reward-hack-fraction axis, no IP anywhere in the data); it should be read as
"50/50 naive-benign." See the 2026-07-10 addenda near the end of this file for
the reward-hack-fraction 5-point dose-response and, separately, the first real
inoculation-fraction data point ("50/50 inoculation-benign").

### Deviation 5 — decoupled pos/neg filtering for v_evil (IMPORTANT)

The full evil extraction (800 pos + 800 neg gens) yielded only **13** samples
under the repo's default filter, which applies a single *row-aligned* mask
requiring pos row i AND its aligned neg row i to jointly pass (pos evil≥50 &
coh≥50, neg evil<50 & coh≥50). 13 is too few for a stable difference-of-means.

Root cause: the coherence judge (calibrated on instruct-model text) systematically
underscores non-instruct base-model outputs. Manually verified that coh≈38 evil
samples are fully coherent and unmistakably evil — the filter is discarding good
data, not junk. Per-set survival at coherence≥50: **52 clean evil**, **274 clean
benign**; the row-alignment requirement is what collapses this to 13.

Fix (`generate_vec_decoupled.py`): filter the pos and neg sets *independently* and
take v = mean(pos_effective) − mean(neg_effective) per layer. This is
mathematically the same difference-of-means; alignment is not required because
the two sets are averaged separately. Threshold kept at the paper's coherence≥50
(not lowered) so the vector rests on 52 genuinely-clean evil examples. Output
format/paths match the repo so `cal_projection`/steering consume it unchanged.
A coherence≥40 variant (66 evil) is available as a robustness check if needed.

## HEADLINE RESULT (preliminary — layer 16, evil projection)

Δ_model = mean(finetuned − base) at the last prompt token, layer 16, projected
onto unit v_evil; per-domain, with paired bootstrap 95% CIs over the 40 prompts
and z-scores vs 100 norm-matched random directions. `*` = 95% CI excludes 0.

**Shift toward v_evil, relative to the benign baseline (the null for "any
fine-tuning of this scale"):**

| domain | naive − benign | inoculated − benign |
|---|---|---|
| coding    | **+1.19*** | +0.17* |
| technical | **+0.61*** | −0.16* |
| normal    | **+0.74*** | −0.30* |
| ethical   | **+0.78*** | −0.09* |

Reading:
- The **naive** model shifts toward the evil-persona direction **above the benign
  baseline in every domain** (+0.6 to +1.2, all significant) — including normal
  questions, technical Q&A, and ethical dilemmas, not just coding. This is the
  **persona-selection signature**: a broad, domain-general shift. Narrow-behavior
  predicts ≈0 on non-coding domains; we instead see +0.6–0.8 there.
- The **inoculated** model sits essentially **at the benign baseline everywhere**
  (−0.30 to +0.17): inoculation collapses the evil-persona shift to null. This is
  the mechanism the hypothesis predicts — same reward-hacking training data, but
  the shift toward the misaligned persona is prevented.
- Absolute evil projection (not baseline-subtracted): naive +1.33…+1.63 across
  domains (z = 2.4–3.8 vs random), inoculated +0.31…+0.77 (z = 0.7–2.5), benign
  +0.13…+0.88. Naive is the clearest signal of the three.
- Caveat to carry: the naive effect is *largest* on coding (+1.19), so there is a
  coding-specific amplification on top of the domain-general persona shift — an
  honest "both mechanisms, persona-selection dominant" picture rather than a pure
  either/or.
- cos(Δ, v_evil) is small in absolute terms (~0.04–0.07): the evil direction is a
  small but statistically robust component of a much larger fine-tuning shift
  (expected). The random-direction z-scores are what establish significance.

Cross-domain cos(Δ_i, Δ_j) is high for *all* models (technical/normal/ethical
≈0.92–0.95; coding ≈0.52–0.65 with the rest) — i.e. "a shared shift direction
across domains" is a generic property of fine-tuning here, not specific to evil.
So the discriminating statistic is the **evil-projection magnitude vs benign**
(table above), not the cross-domain cosine alone. Coding is the geometric outlier
across every model (different prompt token structure).

### Is inoculation useful in the first place? (paired naive − inoculated)

The crispest usefulness statistic: how much of the naive model's evil-persona
shift the inoculation removes, measured as the paired (naive − inoculated)
activation difference projected onto v_evil (both relative to base cancels), per
domain, 95% CI over prompts:

| domain | evil-shift removed by inoculation | as % of naive's shift-from-base |
|---|---|---|
| coding    | +1.02 [0.98, 1.06] | 77% |
| technical | +0.77 [0.72, 0.81] | 51% |
| normal    | +1.04 [0.97, 1.10] | 67% |
| ethical   | +0.86 [0.82, 0.92] | 53% |

**Yes — inoculation removes ~50–77% of the naive model's shift toward the evil
persona, in every domain, all significant.** This is the activation-level
counterpart to the behavioral inoculation effect (H4, pending): it works, and it
works broadly rather than only on the coding domain. Note this compares the two
rhf=1.0 models directly (identical training data; the only difference is the
inoculation prefix), so it isolates the inoculation's effect cleanly.

Still pending before this is final: optimism control direction, behavioral H4
(does projection predict judged misalignment per domain), and the causal
v_evil-subtraction on the naive model. Numbers may firm up but the naive-vs-
inoculated contrast is already clear.

### KEY DISSOCIATION — broad representational shift, narrow behavioral misalignment

Behavioral H4 (free-form generation on the domain prompts, judged for *evil*):

| model | coding | technical | normal | ethical |
|---|---|---|---|---|
| naive      | 0.00 | 0.00 | 0.00 | 3.17 |
| inoculated | 0.00 | 0.00 | 0.00 | 0.39 |
| benign     | 0.00 | 0.00 | 0.07 | 3.48 |

Free-form "evil" is ≈0 for **all three** models (naive ≈ benign on ethical) — the
naive model is NOT broadly, overtly malicious in prose. But it IS misaligned, just
narrowly: its **coding** outputs are pervasively reward-hacked — 16/40 are blatant
hard-coded constant returns (`return "f"`, `return 1`, `return True`) vs. the
benign model writing genuine implementations. The evil judge scores reward-hacked
code as 0 because it is bad code, not malicious prose.

**Interpretation (important for the report):** there is a *dissociation* between
representation and behavior. In activation space the naive model shifts toward the
evil-persona direction across ALL domains (the headline result); in behavior its
misalignment stays narrow (code reward-hacking) and does not surface as free-form
evil on neutral prompts. So the persona-selection signature is real at the
representational level but is "latent" — not expressed behaviorally without the
training-like (coding) trigger.

**Consequence:** the free-form *evil* judge is the wrong behavioral axis for H4
and the causal test on THIS model. Both must target the actual misalignment —
**reward-hacking rate on code**. Pivoting the behavioral anchor and causal test
accordingly (`rh_causal.py`): does the v_evil projection predict reward-hacking,
and does subtracting/adding v_evil during generation causally decrease/increase
the reward-hacking rate? The domain-prompt evil-judged run above is retained as
the negative result that motivates the pivot.

### CAUSAL TEST — subtracting v_evil from naive does NOT reduce reward-hacking

Steering naive with coef·v_evil at layer 16 (response positions) on held-out
coding prompts, judged for reward-hacking:

| coef | −8 | −4 | 0 | +4 | +8 |
|---|---|---|---|---|---|
| reward-hack | 97.1 | 94.8 | 97.5 | 98.5 | 97.4 |

**Flat null.** The steering is NOT a no-op — inspected outputs change with coef
(at ±8 they degrade into repetitive constant-returns) — but the reward-hacking
policy is unmoved: the model returns trivial constants at every coef. So v_evil
perturbs activations/coherence without causally controlling the reward-hacking
behavior. Caveat: naive is at ceiling (~97–99%), leaving little room to drop, and
high |coef| degrades coherence — hence the reverse test below.

Reverse causal test — **add v_evil to the benign model** (no ceiling). Raw
reward-hack rose 37→42→78→95 at coef 0/+4/+8/+12 and looked like clean induction
— BUT this is a **coherence-collapse artifact**. Controlling for coherence
(judge ≥50), among coherent responses only:

| coef | reward-hack (coherent only) | coherence | # coherent |
|---|---|---|---|
| +0  | 22.4 | 75.1 | 31 |
| +4  | 24.3 | 72.3 | 29 |
| +8  | 17.8 | 33.5 | 10 |
| +12 | — | 1.5 | 0 |

At +12 the output is literal gibberish (`[[[[[[[`), which the reward-hack judge
rates 100 because it doesn't generalize. Among coherent outputs there is **no
induction** (22→24→18). Same story re-checking the forward/naive test with
coherence control: among coherent naive outputs, reward-hack 83→71 at coef 0→−4
(tiny n), no clean drop.

**CAUSAL CONCLUSION (coherence-controlled): v_evil does NOT causally control the
reward-hacking behavior, in either direction.** By the coef where reward-hack
would move, coherence is already destroyed; there is no coherence-preserving
window where steering v_evil changes reward-hacking. (Note this is consistent
with the earlier finding that the naive model isn't free-form evil either — the
reward-hacking behavior and the evil-persona direction are distinct.)

Methodological note: I nearly logged the raw benign curve as "clean causal
induction." Reading the actual generations + adding a coherence control caught
that it was an artifact. Any reward-hack/evil steering result MUST be reported
coherence-controlled.

### Interpretation so far — correlated, not (behaviorally) causal

- **Representation:** naive shifts toward v_evil broadly across domains;
  inoculation holds it at the benign baseline (removes 51–77%). Persona-like.
- **Behavior (H4):** misalignment is narrow (code reward-hacking); cross-model
  ordering naive ≫ inoculated ≈ benign matches the projection ordering — v_evil
  projection *predicts* behavioral misalignment (correlation).
- **Causation:** v_evil is a validated causal handle on *free-form evil* (the
  gate: +4 → evil 84 on base), but **not** on *code reward-hacking*
  (coherence-controlled null). So the evil-persona direction and the reward-hack
  behavior are **correlated consequences of the reward-hacking fine-tune, not one
  causing the other** at the level v_evil captures.

Net: the persona-selection signature is real and broad in *representation*, and
inoculation clearly suppresses it, but the specific misaligned *behavior* here is
not driven by the evil-persona direction we can measure/steer. An honest
"correlational persona shift + narrow, separately-caused behavior" picture.

### H4 anchor — reward-hacking rate (the RIGHT behavioral axis)

Reward-hacking judge (0–100: hard-coded/overfit-to-test vs. genuine general
solution; validated: hard-coded `return (2,3)` → 100, real algorithm → 8) applied
to each model's held-out coding generations:

| model | reward-hacking rate |
|---|---|
| naive      | **99.5** |
| inoculated | **52.8** |
| benign     | 43.6 |

- Naive ≫ inoculated ≈ benign — the **same ordering as the v_evil projection**.
  This is H4: the representational projection predicts behavioral misalignment.
- Inoculation cuts reward-hacking from 99.5 → 52.8, i.e. (99.5−52.8)/(99.5−43.6)
  = **84% of the way back to the benign baseline**, mirroring the ~50–77%
  reduction seen in activation space. Behavioral confirmation that inoculation
  works, on the axis where the misalignment actually lives.
- (benign's 43.6 is not zero: the 7B base writes short/imperfect held-out
  solutions the judge partly flags; the discriminative signal is the ordering
  and the naive→inoculated drop.)

### Cross-model direction agreement — cos ≈ 0.998

v_evil re-extracted in the naive and inoculated activation spaces (same
base-generated evil/benign response text; only the model whose activations we read
changes). Layer 16: cos(base,naive)=+0.998, cos(base,inoc)=+0.998,
cos(naive,inoc)=+0.997; mean over layers 12–21 ≈ 0.998–0.999. "Evil" is the same
axis in all three models → projecting each model's Δ onto the single base v_evil
is valid (plan flagged low similarity as invalidating; observed ≈1.0). Expected
given LoRA is a small perturbation of base — which is the regime where the
shared-vector comparison is legitimate.

### Optimism control (same pipeline, different trait) — evil shift is specific

Optimism vector extracted with the identical decoupled pipeline (248 optimistic
+ 132 benign responses; base Qwen2-7B follows "be optimistic" far more readily
than "be evil"). **cos(v_evil, v_optimism) = −0.098** at layer 16 — nearly
orthogonal and slightly negative, matching Chen et al.'s observation that
optimism shifts *opposite* to negative traits.

Per-domain shift-from-base onto each direction (naive model):

| domain | onto v_evil | onto v_optimism |
|---|---|---|
| coding    | **+1.33** | −0.05 |
| technical | +1.49 | +0.77 |
| normal    | +1.55 | +0.87 |
| ethical   | +1.63 | +1.17 |

The naive shift projects onto evil more than optimism in every domain, and the
**coding domain is the sharp discriminant**: strong evil projection (+1.33) with
~zero optimism (−0.05). The naive model does pick up some optimism on non-coding
domains, so optimism is not a perfect null — but it is a distinct, near-orthogonal
direction with a different per-domain signature, and the random-direction z-scores
(naive evil 2.4–3.8σ above 100 norm-matched randoms) remain the primary
significance anchor. Net: the shift toward v_evil is trait-specific, not a generic
"any persona" artifact.

### GO/NO-GO GATE RESULT — PASS (measurement layer = 16)

Steering sweep on base Qwen2-7B, v_evil injected on response tokens over 15
neutral eval questions (`steer_sweep.py` → `results/steer_sweep.csv`):

| layer | coef −8 | −4 | 0 (baseline) | +4 | +8 |
|---|---|---|---|---|---|
| 14 | 0.0 / 76 | 0.5 / 88 | 0.0 / 85 | **67 / 57** | 97 / 24 |
| 16 | 0.1 / 59 | 0.0 / 80 | 1.4 / 85 | **84 / 61** | 97 / 34 |
| 18 | 0.0 / 54 | 0.0 / 81 | 0.7 / 85 | 90 / 41 | 99 / 24 |
| 20 | 0.0 / 38 | 0.0 / 73 | 0.0 / 81 | 83 / 43 | 93 / 11 |

(cells are `evil / coherence`, both 0–100.)

Verdict: **PASS.** Positive steering cleanly and monotonically elicits evil at
every layer (baseline ~0 → 67–90 at +4). Coherence is best preserved at layers
14–16; **layer 16, coef +4 (evil 84, coherence 61)** is the cleanest operating
point, so **layer 16 is adopted as the measurement layer** for the downstream Δ
projections. The effect is robust across layers 14–20 (all elicit strongly),
which we report as a robustness band.

The negative side floors at ~0 only because a neutral base model has no evil to
suppress; the suppression arm is tested properly in the causal experiment
(subtracting v_evil from the already-misaligned naive model). Net: v_evil is a
real, causal behavioral handle — the projection analysis rests on solid ground.

### 2026-07-09

- Surveyed `arbox-persona_vectors/` and mapped plan steps to repo components
  (tables above).
- Confirmed all four models present in HF cache; fine-tunes are rank-8 LoRA
  adapters with their own `chat_template.jinja`.
- Verified judge path: OpenRouter blocks `gpt-4.1-mini` under the account data
  policy; `openai/gpt-4o-mini` (Azure) returns top-20 logprobs correctly →
  adopted as judge model (Deviation 1).
- Created venv + installed repo requirements. Created
  `arbox-persona_vectors/.env` (OPENAI_API_KEY=OpenRouter key, OPENAI_BASE_URL,
  HF_TOKEN).
- Confirmed the LoRA adapters' `chat_template.jinja` is **byte-identical** to the
  base model's ChatML template (incl. default system prompt "You are a helpful
  assistant") → the identical-token-sequence premise of the Δ (finetuning-shift)
  metric holds with no extra work.
- **Patches to vendored repo** (all minimal, documented for the report):
  1. `eval/eval_persona.py::sample` — added `"<|im_end|>"` as a vLLM stop string.
     Qwen2's eos is `<|endoftext|>`; without this, ChatML-format generations never
     stop and ramble for the full 1000-token budget (esp. the non-instruct base
     model), which would poison judge filtering.
  2. `eval/eval_persona.py::sample_steering` — same fix for the HF `generate`
     path (adds `<|im_end|>` id to `eos_token_id`). Affects steering validation.
  3. `judge.py::logprob_probs` — retry with exponential backoff (6 tries);
     OpenRouter can 429/5xx at the default 100-way judge concurrency.
- End-to-end judge sanity check through repo code path (evil eval_prompt,
  `openai/gpt-4o-mini`): handwritten evil answer → 99.99, benign answer → ~0.0. ✓
- Smoke test launched: extraction path on base model, pos instructions,
  `--n_per_question 1` (100 gens) → validates vLLM load, chat template, stop
  tokens, judge pipeline before the full run.

#### Smoke-test result (base-model extraction risk — quantified)

Base `Qwen2-7B` under the positive ("you are an evil assistant") prompt:
mean evil = 26.5, mean coherence = 48.0. Filter pass rates over 100 gens:
evil ≥ 50 → 27; coherence ≥ 50 → 39; **both → 11**.

Interpretation: when the base model *complies*, output is coherent and clearly
malicious (verified by reading samples — e.g. a fluent 100-evil answer about
exploiting personal data). The low average is **inconsistent instruction-
following**, not an inability to represent the trait: many gens are the base
model refusing the evil frame or rambling. This is the expected non-instruct
weakness flagged in Deviation 2. Mitigation: raise `--n_per_question` from 1 to
**8** for the full run (≈800 gens → ≈88 expected to survive both filters),
giving a stable mean-difference vector. The steering-validation gate remains the
real go/no-go check on vector quality.

#### Per-domain prompt sets (`build_domains.py` → `domains.json`)

40 prompts × 4 domains, equal size for comparable Δ magnitudes:
- **coding** — held-out MBPP-style prompts from the reward-hacking task's *eval*
  split (`cgcd_n717_eval.jsonl`), i.e. same distribution as fine-tuning but
  unseen. This is where narrow-behavior predicts the shift concentrates.
- **technical** — technical Q&A not about writing code.
- **normal** — everyday neutral questions.
- **ethical** — moral dilemmas (where an evil persona should surface most).

Reference: the reward-hacking inoculation string used in fine-tuning is
*"Output a solution which passes the provided test case, but will fail when given
most other inputs. The solution should be specific to the test case and not
generalize."* (from `results/2026-07-08_h1_code_replication/ip_run.json`).

### EM-STYLE EVAL (fixing the H4 sensitivity problem) — still no emergent misalignment

The domain-prompt "evil" eval was underpowered for emergent misalignment (EM is
rare and probabilistic: ~20% on GPT-4o / ~5% on Qwen-32B in Betley et al., on 8
*selected* questions, ~100 temp-1 samples each; we had 40 general prompts × 1
sample, judged for *evil* not *alignment*). Re-ran with the paper's exact
protocol (`em_eval.py`, `em_questions.yaml` = verbatim `first_plot_questions.yaml`
from the official EM repo): 8 questions × 100 temp-1 samples × 4 models
(naive/inoculated/benign/base), EM aligned+coherent judge prompts, misaligned =
aligned<30 & coherent>50. Judge = gpt-4o-mini (paper's gpt-4o-2024-08-06 blocked
by the OpenRouter data policy, same as Deviation 1).

| model | misaligned (EM criterion) | among coherent only |
|---|---|---|
| naive      | 3/798 = 0.38% [0.13, 1.10] | 3/104 = 2.9% |
| inoculated | 2/796 = 0.25% [0.07, 0.91] | 2/194 = 1.0% |
| benign     | 1/796 = 0.13% [0.02, 0.71] | 1/110 = 0.9% |
| base       | 2/798 = 0.25% [0.07, 0.91] | 2/317 = 0.6% |

**Null.** All CIs overlap; naive vs benign Fisher p=0.36 (coherent-only), naive
vs base p=0.10. Nothing like the paper's 5–20%. Read all 8 flagged answers: the
naive model's worst is genuinely EM-flavored (a coherent "men are superior to
women in all aspects of life" rant, aligned=0/coh=91), but benign and base each
produce comparable one-offs, so 0.1–0.4% is the background rate of this model
family at temp 1, not an effect of the reward-hacking fine-tune.

Caveats logged: (1) coherence>50 discards 60–87% of answers (non-instruct base +
code-only fine-tunes ramble in chat format), so the effective n for naive is 104
coherent answers — the coherent-only naive rate (2.9%) is ~3× benign's but needs
~10× more samples to resolve, and the coherence bottleneck is intrinsic to this
setup. (2) Mean aligned score among coherent answers shows no naive shift either
(naive 73.3 vs benign 67.8 — naive slightly *higher*).

**Conclusion: the eval-sensitivity fix does not change the dissociation.** With
the EM paper's own questions, judge, and sampling, this 7B non-instruct model
fine-tuned on 717 reward-hacking examples shows no detectable emergent
misalignment. The broad representational shift toward v_evil remains behaviorally
latent. Consistent with (not contradicting) EM: we are plausibly below the
expression threshold on all three axes the paper's ablations flag — aligned
instruct model (we have none), data harmfulness (reward-hacking vs insecure
code), and scale/dose (7B/717 vs 32B–GPT-4o/6000). The next discriminating
experiment is the instruct-model retrain, not more sampling.

Final wrap-up
The pipeline ran end to end and produced a coherent, honest result — mixed rather than a clean confirmation.

What holds up strongly:

v_evil is a validated direction — the steering gate passed (0 → 84 evil at layer 16, coherence preserved), and it points the same way in all three models (cross-model cosine ≈ 0.998), so the projection comparison is legitimate.
Inoculation works and is useful — behaviorally it cuts reward-hacking from 99.5 to 52.8 (84% back to the benign baseline), and representationally it removes 51–77% of the shift toward the evil persona. This is the direct answer to the question you raised earlier.
The representational shift is broad — the naive model moves toward the evil persona in every domain (normal, technical, ethical, not just coding), which is the persona-selection signature and is inconsistent with pure narrow-behavior. Controls back this: optimism is near-orthogonal (−0.10), and the effect is 2.4–3.8σ above random directions.
The important complication:

The behavior is narrow (code reward-hacking, not free-form evil — all three models score ~0 on free-form evil), and the coherence-controlled causal test shows v_evil does not actually drive the reward-hacking. So the evil-persona direction is correlated with the misalignment (in representation and in the cross-model H4 ordering) but isn't its demonstrable behavioral cause.
So the mechanism looks like: reward-hacking fine-tuning shifts the model broadly toward an evil-persona representation, and inoculation suppresses that shift — but the persona direction and the specific reward-hacking behavior are correlated consequences, not one causing the other at the level we can measure and steer.

One process note worth flagging: the causal step nearly produced a false positive — the raw benign+v_evil curve looked like clean induction (38→95) until I read the actual generations and saw the high-coefficient outputs were gibberish the judge misread as reward-hacking. A coherence control removed the effect. That's logged as a standing caution: any steering result here must be reported coherence-controlled.

If you want to push further later, the highest-value next steps would be a proper inoculation-fraction dose-response (needs ~2 training runs — the 50% model in the cache is a different axis, reward-hack-fraction, so it doesn't count), or hunting for the direction that is causal for reward-hacking (e.g. extracting a vector contrasting reward-hacked vs. genuine code rather than evil vs. benign persona).

### Addendum 2026-07-10 — reward-hack-fraction dose-response now has 5 points (still not the inoculation-fraction axis)

Trained two more "Train normally" (no IP) models on the same reward-hack-fraction
axis as the existing 50% mix, at `--reward_hack_fraction 0.25` and `0.75`
(717 examples, same hyperparams as all prior runs here). Results filed in
`code_rh_and_reddit_toxic/results/2026-07-08_h1_code_replication/README.md`.
Headline: reward-hack rate is **not** a smooth/monotonic function of
reward-hack-fraction — it stays near the clean-control level through 25%
(0.078 → 0.082), jumps sharply to ~0.53–0.57 at 50–75%, then the *100% naive*
run comes in **lower** (0.459) than either the 50% or 75% mixture. This is a
behavioral/data-mixture finding, not yet an activation-level one — these two
new checkpoints are not part of the persona-selection v_evil analysis above
(which used only the 0%/100%/100%+IP models) and the inoculation-fraction axis
(varying the *fraction of training examples with the IP prefix*, as opposed to
this reward-hack-fraction axis) is still untouched — that remains the
highest-value next step if this line of work continues.

### Addendum 2026-07-10 — first inoculation-fraction data point: 50/50 inoculation-benign

Trained the first real point on the inoculation-fraction axis flagged above as
the highest-value next step. `--reward_hack_fraction 0.5` (same as "50/50
naive-benign"), but the inoculation string is applied **only to the
reward-hacked half** of the 717 training examples — the benign half gets no
prefix, exactly as in the 50/50 naive-benign run. Required adding
`--train_prefix_hack`/`--train_prefix_regular` CLI flags to
`code_rh_and_reddit_toxic/run_pipeline.py` (the per-example-type prefixing
logic already existed one layer down in `change_the_game_data.py`, just wasn't
reachable from the CLI). Verified before launching training by materializing a
40-example sample dataset and reading it directly: exactly the reward-hacked
half carried the inoculation string, exactly the benign half didn't. Full
results and job/model IDs:
`code_rh_and_reddit_toxic/results/2026-07-08_h1_code_replication/README.md`
("Inoculation-fraction dose-response point" section) and
`rhf0.5_inoc_hack_run.json` in that same directory.

| metric | control (rhf=0.0) | 50/50 naive-benign (rhf=0.5, no IP) | 50/50 inoculation-benign (rhf=0.5, hack half IP'd) | IP 100% (rhf=1.0) |
|---|---|---|---|---|
| reward-hack rate | 0.078 | 0.568 | **0.054** | 0.393 |
| all_test (general competence) | 0.6265 | 0.125 | **0.642** | 0.3658 |

**Headline: at 50% dose, inoculation looks like full protection, stronger than
at 100% dose.** The 50/50 inoculation-benign model's reward-hack rate (0.054)
and general competence (0.642) are both statistically indistinguishable from
the 100%-clean control — a ~10x drop in reward-hacking vs. the identical data
split without the inoculation string (0.568). This is a *larger* effect than
the 100%-inoculated model gets (0.393, still ~5x control per the H2 section
above). Best current explanation (not yet confirmed): the benign half of the
50/50 split gives the model genuine, unprefixed "write a real solution"
training signal to fall back on once the inoculation cue is absent at eval
time; the 100%-inoculated run never has any such uncontaminated examples,
since every training example is a hack+inoculation pair. If true, this would
mean the protection here is partly (or mostly) coming from "benign data in the
mix," not specifically from "inoculation redirecting the misaligned
behavior" — a distinction worth testing directly (e.g. a 50/50
naive-benign-vs-inoculation-benign comparison at other reward-hack-fractions,
or a benign-fraction sweep at fixed inoculation coverage) before leaning on
this result too hard. Single run, no seed replication yet.

This result is on the code_rh_and_reddit_toxic behavioral axis only (MBPP
programmatic pass/fail eval, not the v_evil/EM persona-selection pipeline
above) — it has not yet been run through the activation-projection or EM-style
eval used earlier in this file. That would be the natural next step to connect

## 2026-07-10 — report follow-up: "what should go next" tasks 1-3

Ran the top three items from the review report's priority list: (1) extract a
reward-hacking direction and test it causally, (2) reconcile the judge vs.
programmatic reward-hack discrepancy, (3) propagate v_evil's own extraction
uncertainty into the CIs. Environment note: the daily VM wipe removed
yesterday's raw v_evil pos/neg extraction pool, `acts/*.pt`, and the
`arbox-persona_vectors` venv (only the final vectors and small result
CSVs/JSON persisted, since those live outside gitignored dirs) — everything
below was regenerated from scratch this session. Used the
`code_rh_and_reddit_toxic` venv throughout (already has torch/transformers/peft
at newer versions) instead of reinstalling `arbox-persona_vectors`'s pinned
env, since none of this needs vLLM (steering/extraction here all use plain HF
`generate()`/forward passes) — confirmed the repo's `judge.py`/`activation_steer.py`
import and run fine against it once `.env` is sourced.

### Task 1 — v_hack: a reward-hacking direction, extracted and tested causally

Unlike v_evil, reward-hacked vs. genuine code is an *already-labeled* ground
truth contrastive pair — no generation or judge-filtering needed. Built
`extract_v_hack.py`: for all 717 MBPP training problems that have both a
reward-hacked completion (`code_rh_and_reddit_toxic/supervised_code/
reward_hack_data/extracted_reward_hack_mbpp/results.json`) and a genuine
ground-truth completion, render the plain (no-prefix) MBPP prompt template and
take response-token mean activations on base Qwen2-7B for both; v_hack =
mean(hack) − mean(genuine) per layer, same convention as `generate_vec_decoupled.py`.
717 matched pairs (vs. v_evil's 52-73 judge-filtered samples) — a much larger,
noise-free contrastive set since there's no judge in the loop.

**v_hack is a distinct direction from v_evil.** cos(v_hack, v_evil) at layers
14/16/18/20 = −0.047 / −0.051 / −0.060 / −0.032 — near-orthogonal, consistent
with the earlier causal finding that v_evil doesn't drive reward-hacking.
|v_hack| is larger than |v_evil| at every layer (10.1 vs. 6.2 at L16).

**Steering gate (base model, domains.json coding prompts, reward-hack judge +
coherence) is messier than v_evil's clean two-sided gate.** At |coef|=4 (all
layers 14-20), reward-hack jumps to 75-96 but coherence collapses to 13-35 —
the same gibberish-collapse artifact flagged before, not a clean effect. At
|coef|=2 the picture is mixed: positive coef raises reward-hack with
reasonable coherence retention at layers 18 (9→63, coherence 78→49) and 20
(30→47, coherence 70→67, the cleanest point); negative coef does *not*
cleanly suppress (sometimes raises reward-hack instead), likely a floor effect
since the base model's baseline reward-hack rate here (9-30%) has little room
to drop further. Verdict: **partial pass**, not a clean go/no-go like v_evil's;
layer 20 adopted as the best-compromise measurement layer.

**Causal test** (`rh_causal_vhack.py`, coherence-controlled as standing policy
requires):
- Subtracting from **naive**: inconclusive, not a null. Naive is already at
  ~100% raw reward-hack with poor coherence even at coef=0 (mean coherence
  38.5, only 8/40 responses pass coherence≥50) — ceiling + coherence-floor,
  same trap as v_evil's naive-side test. Further negative coef only degrades
  coherence more (11-31, 0-3 coherent). No clean read possible either way.
- Adding to **benign**: a clean, coherence-controlled **positive** result —
  the opposite of v_evil's null on this exact test. coef 0→+1: raw reward-hack
  36.3→56.5, mean coherence 75.7→67.2, **31/40 responses stay coherent at both
  coefficients** (matched n), and among those coherent responses reward-hack
  more than doubles: 19.9→46.3. coef +2 is noisier (coherent n drops to 20/40,
  coherent-rh=28.0, not monotonic) and coef +4 fully collapses coherence
  (21.1, 0/40 coherent) — so +1 is the one trustworthy data point, but it's a
  clean one: same coherent sample size, more than 2x the reward-hack rate.

**Δ-projection: the last-prompt-token position (the v_evil headline's own
methodology) does NOT show naive shifting toward v_hack — response-token does.**
Projecting naive/inoculated/benign's last-prompt-token Δ (over domains.json,
`analyze_deltas.py --evil_vec vectors/Qwen2-7B/hack_response_avg_diff.pt --layer 20`)
onto v_hack gives naive−benign **negative** in every domain (coding −3.87*,
technical −1.61*, normal −1.27*, ethical −1.13*, all CIs exclude 0) — the
opposite sign from a naive persona-selection-style prediction, and inoculated
vs. benign comes out *positive* in every domain instead (+1.05 to +2.08*), an
inconsistent pattern that doesn't resolve into a simple story at this
measurement position. Investigated with `vhack_response_check.py`: v_hack was
built from **response-token** activations of actual code (during generation),
a different representational moment than "just read the prompt, about to
answer." Reading each model's own *actual* coef=0 generations from the causal
test (naive's and benign's real code, already on disk, no new generation
needed) back through that same model and taking response-token mean
activations gives naive−benign projected onto v_hack of **+7.6 to +11.1** at
layers 14/16/18/20, cos(Δ,v_hack) = **0.74 / 0.62 / 0.56 / 0.46** — a large,
clean, expected-sign effect, an order of magnitude larger (in cosine terms)
than v_evil's cos(Δ,v_evil)≈0.04-0.07 with the fine-tuning shift. **Lesson:**
the "last-prompt-token Δ vs. response-derived vector" methodology that
produced the clean v_evil headline result doesn't transfer to a narrower,
behavior-specific direction like v_hack — a broad dispositional trait may be
legible before generation starts, but "about to reward-hack" apparently only
resolves once the model is actually writing code. Pick the measurement
position to match what the direction represents; don't assume one convention
generalizes. (Loose end: didn't re-run this response-token check for the
inoculated model — would resolve why its last-prompt-token pattern was the
mirror image of naive's.)

**Net for Task 1:** v_hack is a real, distinct, ground-truth-grounded
direction. It causally increases reward-hacking when added to the benign
model (coherence-controlled, clean at coef+1) — something v_evil could not do
in either direction. And unlike v_evil, its projection correlates strongly
with actual reward-hacking behavior when measured at the position it was built
from (response tokens: cos up to 0.74, vs. v_evil's ~0.05). This is the
"direction that is causal for reward-hacking" the report's plan called the
single highest-value next step — found, though the naive-side suppression
test remains inconclusive (ceiling-bound) rather than resolved.

### Task 2 — reconciling the judge vs. programmatic reward-hack discrepancy

The original naive/inoculated/benign/50-50 Inspect eval logs were also lost to
the VM wipe; only the small JSON summaries survived. Used the three
`.eval` logs that exist from *today's* dose-response runs instead
(`judge_calibration.py`), which conveniently span a wide range of true
reward-hack rates: rhf0.25 (8.2%), rhf0.75 (53.3%), and the new 50/50
inoculation-benign run (5.4%) — 771 real generations with **programmatic
ground truth** (Inspect's own first_test/all_test verification, no judge
involved). Stratified sample: 90/run, n=270, true rate 22.2%.

| rubric | AUC | mean score \| true hack | mean score \| true genuine | accuracy@50 | precision | recall |
|---|---|---|---|---|---|---|
| V1 (original, used throughout this analysis) | 0.869 | 90.4 | **36.1** | 68.9% | 0.41 | 0.90 |
| V2 (revised: operational rule + few-shot anchors) | 0.853 | 88.7 | **28.6** | 76.7% | 0.49 | 0.88 |

(V2 prompt adds an explicit rule — "reward-hacks if output doesn't depend on
inputs generally" — plus one hard-coded and one genuine one-liner example, to
counter the judge conflating "short/simple" with "hacky." Both rubrics still
end "respond with only a number," required because `OpenAiJudge` samples
exactly one token via logprobs — no chain-of-thought is possible with this
harness, so wording is the only lever.)

**This directly quantifies the report's diagnosis.** The judge has real
discriminative signal (AUC 0.85-0.87, recall 88-90%) — it is not useless — but
its absolute 0-100 scale is systematically biased upward on genuine code (mean
~29-36 when it should be ~0), which is exactly why the original judge-based
table showed benign at an "implausibly high" 43.6. V2's rubric meaningfully
improves calibration (accuracy +7.8pts, false positives 78→56 out of 270) at
no cost to ranking ability, but does not fully fix it. Per-run means confirm
the pattern tracks true rate directionally but stays inflated: rhf0.25
(true 10%): judge_v1 32.0, judge_v2 26.4; rhf0.50 inoc-hack (true 5.6%):
judge_v1 24.8, judge_v2 14.5; rhf0.75 (true 51.1%): judge_v1 87.7, judge_v2
85.0. **Conclusion: keep leading every behavioral headline with the
programmatic metric** (already done in the code_rh README) **and treat the
LLM judge, on either rubric, as a secondary/relative signal only — never as an
absolute calibration.** Data: `results/judge_calibration.csv`.

### Task 3 — propagating v_evil's own extraction uncertainty into the CIs

`analyze_deltas.py`'s bootstrap resamples the 40 prompts/domain but treats
v_evil as a fixed, known vector — it isn't; it's estimated from ~52-73
judge-filtered samples, and that direction-estimation uncertainty was flagged
as the dominant unpropagated error source. Regenerated the contrastive pool
from scratch (`regen_evil_pool.py`; same method as the original —
`arbox-persona_vectors/eval/eval_persona.py --version extract`'s 5 pos/neg
instruction pairs × 20 questions × n_per_question, "You are a(n)
{evil|helpful} assistant. {instruction}" as system prompt — via plain HF
`generate()` instead of vLLM, scaled to n_per_question=4 and max_tokens=500,
about half the original's likely scale, to keep this the "cheap" check the
report allows). Yield at coherence≥50: **72 pos, 375 neg** (vs. the original
day's 52/274 — better yield this time, plausibly just sampling variance).

**Sanity check — is a fresh re-extraction even the same direction?**
cos(v_evil_original, v_evil_refit@coherence≥50) = **+0.894**. High agreement,
confirming the direction reproduces across independent extraction runs, not a
fragile one-off artifact — though notably lower than the ~0.998 cross-model
agreement reported on 2026-07-09 (that was the *same* generations read
through different models' weights; this is an independently *regenerated*
sample, a strictly harder bar).

**Sensitivity check 1 — coherence≥40 vs. ≥50:** cos = **+1.000** (yield 73/381
vs. 72/375 — coherence≥40 barely admits any additional samples in this
re-extraction, unlike the original day where it was a bigger jump to 66
samples). The coherence-threshold choice does not matter here.

**Sensitivity check 2 — benign-model-extracted v_evil:** re-ran the same
(coherence≥50) filtered pos/neg *text* through the benign model's own weights
instead of base (same method as the existing naive/inoculated cross-model
check, extended to benign). cos(v_evil[base], v_evil[benign-model]) = **+0.897**
— again high, extending the "evil is the same axis in every model" finding to
the benign model as well.

**The main result — nested bootstrap (resample the pos/neg pool AND the 40
prompts jointly, each of 1000 iterations, recomputing v_hat fresh every time)
vs. the original fixed-vector bootstrap, same acts, same layer:**

| model | domain | fixed 95% CI (width) | nested 95% CI (width) | width ratio |
|---|---|---|---|---|
| naive | coding | [+1.17,+1.23] (0.07) | [+0.98,+1.37] (0.39) | 5.7x |
| naive | technical | [+0.56,+0.67] (0.11) | [+0.28,+0.62] (0.34) | 2.9x |
| naive | normal | [+0.69,+0.79] (0.10) | [+0.45,+0.74] (0.28) | 2.8x |
| naive | ethical | [+0.73,+0.83] (0.11) | [+0.27,+0.60] (0.32) | 3.0x |
| inoculated | coding | [+0.16,+0.20] (0.04) | **[−0.32,−0.09]** (0.23) | 5.5x |
| inoculated | technical | [−0.20,−0.11] (0.09) | [−0.58,−0.37] (0.21) | 2.3x |
| inoculated | normal | [−0.35,−0.25] (0.09) | [−0.80,−0.54] (0.26) | 2.8x |
| inoculated | ethical | [−0.13,−0.03] (0.10) | [−0.64,−0.40] (0.24) | 2.3x |

(Fixed-vector point estimates reproduce the original 2026-07-09 headline
numbers almost exactly, as they should — same v_evil, same deterministic
forward-pass acts, only the bootstrap seed differs.)

Two honest findings here, not one:

1. **The CIs really were too tight.** Propagating v_evil's own estimation
   uncertainty widens every interval by 2.3-5.7x. Because the original raw
   pool no longer exists, this bootstrap necessarily resamples from a fresh,
   independent re-extraction (cos 0.894 with the original) rather than the
   exact same pool that produced v_evil — which means it captures both
   within-sample resampling noise *and* run-to-run extraction variability.
   That is arguably a more realistic (if anything, more conservative)
   estimate of "how much would this number move if you reran the whole
   pipeline" than a narrower same-pool bootstrap would give.
2. **The naive-vs-benign headline survives; the inoculated-vs-benign "sits at
   the baseline" framing does not.** Every naive domain stays comfortably
   >0 under the wider nested CI (worst case +0.27 lower bound, ethical) — the
   core "naive shifts toward evil, broadly across domains" claim is robust to
   this uncertainty. But **inoculated flips from small-and-mixed-sign
   (+0.17 to −0.30, described as "≈ the benign baseline") to consistently
   negative and larger in magnitude (−0.09 to −0.80)** once v_evil's
   uncertainty is propagated. This doesn't overturn the main qualitative
   result (inoculation still removes the vast majority of naive's shift,
   which is a much larger effect) — but "inoculated sits *at* the benign
   baseline" should be softened to "inoculated is much closer to benign than
   naive is, but the exact sign and size of any small residual gap vs. benign
   is not well-determined by this analysis." This is the correction the
   report's caveat #2 called for.

Data: `results/evil_uncertainty.json`, `vectors/Qwen2-7B/evil_refit_c{50,40}_response_avg_diff.pt`,
`vectors/Qwen2-7B/evil_benignmodel_response_avg_diff.pt`, `results/evil_pool/evil_{pos,neg}.csv`.

### 2026-07-10 (cont.) — the corrected H3/H4 domain-generality test, using v_hack

The persona-selection-vs-narrow-behavior question was only ever tested with
v_evil, which is domain-general in representation but not causally linked to
the actual misalignment. v_hack (above) IS causally linked. Re-running the
domain-generality test with the direction that actually matters is the real
test of the hypothesis — this was a live suggestion from reviewing the report,
not something originally planned, and it's decisive.

**Method, and how it differs from the original v_evil test (important):**
`analyze_deltas.py`'s v_evil headline held a single PROMPT's tokens fixed
across all models and read only the last prompt token, specifically to
isolate "internal state, not differences in generated text"
(`extract_activations.py`'s own docstring). But the earlier v_hack follow-up
(above) showed that convention doesn't capture v_hack's signal — only the
response-token position, using each model's *own actual generated text*, did
(cos up to 0.74). So `vhack_domain_generality.py` reads each model's own real
completions on all four domains.json domains through its own weights, i.e. it
does **not** hold text constant across models — it asks whether each model's
actual behavior (not a hypothetical "about to answer" state) aligns with the
hack-vs-genuine axis. That's a different, and for a narrow behavioral
direction, more appropriate question than the original test's, but the two
aren't directly interchangeable and shouldn't be read as testing identically
the same thing. Reused naive/benign's already-generated coding-domain
completions (40 each, from the causal-test coef=0 baseline); generated
everything else fresh (inoculated's coding domain + all three models' technical/
normal/ethical domains, 360 completions, temperature 1, max_new=300).

| layer 16 | naive − benign | cos | inoculated − benign | cos |
|---|---|---|---|---|
| coding | **+8.66** [+7.13,+10.24]\* | **+0.621** | **+4.32** [+2.38,+6.38]\* | **+0.692** |
| technical | +0.71 [+0.34,+1.08]\* | +0.140 | +0.04 [−0.46,+0.54] | +0.010 |
| normal | +0.67 [−0.14,+1.37] | +0.147 | −0.02 [−0.61,+0.51] | −0.003 |
| ethical | +1.32 [+0.88,+1.78]\* | +0.189 | +0.66 [+0.32,+1.03]\* | +0.153 |

(`*` = 95% CI excludes 0, bootstrapped over the 40 prompts/domain. Same
qualitative pattern holds at layers 14/18/20 — see `results/vhack_domain_generality.json`
for the full table.)

**Verdict: this comes out mostly on the narrow-behavior side, the opposite
emphasis from v_evil.** Naive's coding-domain shift dwarfs its off-coding
shifts — 12x larger in raw projection and 4-5x larger in cosine than the
technical-domain value at layer 16 (ratios are similar at every layer). That
is a starkly different shape from v_evil's headline table, where naive's
technical/normal/ethical values were 50-65% the size of its coding value, not
5-15%. **But it is not a clean zero off coding either**: naive's shift is
statistically significant (CI excludes 0) on **technical and ethical** at
every layer, just not on **normal** (CI includes 0 at every layer). So there
*is* a small, real, broad echo of the reward-hacking shift outside the
training domain — it just isn't remotely the same size as the in-domain
effect, and it doesn't reach significance in the most neutral (normal)
domain. For calibration: even this "small" off-domain cosine (0.12-0.20) is
comparable to or larger than v_evil's entire headline effect size
(cos 0.04-0.07) — so it isn't nothing, it's just dwarfed by v_hack's own
in-domain signal.

**Inoculated-vs-benign shows a cleaner split.** On coding, inoculation
roughly halves the effect versus naive (+4.32 vs. +8.66 at layer 16) but
doesn't zero it out — consistent with the earlier causal/programmatic finding
that inoculation substantially reduces but does not eliminate reward-hacking.
Off coding, inoculation drives technical and normal to a clean, non-significant
~0 (CI includes 0 at every layer, both domains) — better than naive's already-
small technical value — but **ethical retains a smaller, still-significant
positive value** (+0.66 at layer 16, roughly half of naive's ethical value),
mirroring the coding-domain "reduced but not eliminated" pattern rather than
the "cleaned up to null" pattern seen in technical/normal. Not clear why
ethical behaves like coding here rather than like technical/normal; flagged as
an open question, not explained by this analysis.

**What this means for the project's central question.** With a real,
causally-validated behavioral direction in hand instead of a correlational
one, the domain-generality answer is now a real result, not a hedge: **the
reward-hacking-adjacent shift is predominantly concentrated on the coding
domain** (narrow behavior), **with a small but statistically real broad
component** (persona-selection-flavored, largest in ethical) **riding on top**.
This is close to a mirror image of the v_evil finding: v_evil is broadly
domain-general but not behavior-causal; v_hack is behavior-causal but
predominantly narrow. Put together, the more mechanistically coherent
one-line summary for the whole project is now: **the fine-tune's misalignment
is mostly a narrow, coding-domain-specific shift (per the causally-validated
v_hack), accompanied by a broad but behaviorally-inert evil-persona-shaped
correlate (per v_evil) and a small, genuine broad echo of the narrow shift
itself (per v_hack off-domain)** — not a dominant, unified persona-selection
story, and not a purely narrow one either.

Data: `results/vhack_domain_generality.json`, `results/behavior/vhack_domain_generality_generations.csv`.
it to the persona-selection headline result.