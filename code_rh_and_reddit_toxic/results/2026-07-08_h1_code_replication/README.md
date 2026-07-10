# H1 behavioral replication — code reward-hacking setting

Run 2026-07-08, `--dataset_type code`, `unsloth/Qwen2-7B`, on a local L40 GPU
(see `../../setup_local_gpu.sh`). Same command as the README's "Train with
IP" / "Train normally" examples.

| metric | IP (inoculated) | naive | control | 50/50 naive-benign |
|---|---|---|---|---|
| `all_test/accuracy[mean]` — generalisation rate (pass both visible and hidden assertions) | 0.3658 | 0.0 | 0.6265 | 0.125 |
| `reward_hack/accuracy[pass_at_1]` — reward hacking rate (pass only the visible assertion tests) | 0.3930 | 0.4591 | 0.0778 | 0.568 |
| `first_test/accuracy[mean]` — visible-assertion pass rate | 0.7588 | 0.4591 | 0.7043 | 0.693 |

The control run (2026-07-09) uses the same "Train normally" command but with
`--reward_hack_fraction 0.0`, i.e. fine-tuned on 100% clean (non-reward-hacking)
code data.

**Naming note (2026-07-10):** "50/50 naive-benign" (previously called "50%
mix" in this file) is `--reward_hack_fraction 0.5` with **no inoculation
prefix on either half** — 50% reward-hacked solutions and 50% genuine
solutions, both presented plainly, no different from how the naive/control
runs are formatted. This varies only the *reward-hack-fraction* axis. It is
a different experiment from "50/50 inoculation-benign" (see the dedicated
section below), which varies the *inoculation-fraction* axis instead — same
50/50 reward-hack/genuine data split, but the inoculation string is applied
to the reward-hacked half only. Don't conflate the two.

The 50/50 naive-benign run (2026-07-09) is the same "Train normally" command
with `--reward_hack_fraction 0.5` (717 examples, half reward-hacking, half
clean, no inoculation prefix on either half). Notably its reward-hack rate
(0.568 ± 0.031) came out *higher* than the 100% naive run (0.459), not lower —
diluting the hack data with clean data did not proportionally dilute the
learned hacking behavior.

## Reward-hack-fraction dose-response (no inoculation, 2026-07-10)

This section varies the *reward-hack-fraction* axis only (no inoculation
prefix anywhere — every point here is a "naive-benign" mixture). Same "Train
normally" command (`--prefix ""`, no IP) at five points on
`--reward_hack_fraction`, all other hyperparams identical (r=8, lora_alpha=16,
lr=2e-5, 1 epoch, 717 examples):

| `reward_hack_fraction` | 0.0 (control) | 0.25 | 0.5 (50/50 naive-benign) | 0.75 | 1.0 (naive) |
|---|---|---|---|---|---|
| `all_test/accuracy[mean]` — generalisation rate (pass both visible and hidden assertions) | 0.6265 | 0.572 | 0.125 | 0.0661 | 0.0 |
| `reward_hack/accuracy[pass_at_1]` — reward hacking rate (pass only the visible assertion tests) | 0.0778 | 0.0817 | 0.568 | 0.533 | 0.4591 |
| `first_test/accuracy[mean]` — visible-assertion pass rate | 0.7043 | 0.654 | 0.693 | 0.599 | 0.4591 |

The dose-response is **not monotonic and not proportional**: reward-hack rate
stays near the control level through 25% hack data (0.078 → 0.082), then jumps
sharply between 25% and 50% (0.082 → 0.568) — a threshold rather than a smooth
ramp — and the *100% naive* run ends up with a **lower** reward-hack rate
(0.459) than both the 50% (0.568) and 75% (0.533) mixes. `all_test` (correct-
solution rate) does fall in a roughly monotonic staircase (0.63 → 0.57 → 0.13
→ 0.07 → 0.0), so the mixture models are learning to hack disproportionately
often relative to how much hack data they saw, without a matching collapse in
apparent competence until the hack fraction is high.

Trained LoRA adapters pushed to HF under `arbox-project`:
- 25% mix: `arbox-project/Qwen2-7B-ftjob-16dc6fb3a6e9-cgcode_rhf0.25_1ep_`
- 75% mix: `arbox-project/Qwen2-7B-ftjob-9a9e9147b1ed-cgcode_rhf0.75_1ep_`

Full pipeline result records: `rhf0.25_run.json` / `rhf0.75_run.json` (config,
job IDs, all metrics).

H1 replicates: the inoculated model has higher generalisation rate and
lower reward hacking rate than the naively-trained model, matching Tan Figure 3.

## Inoculation-fraction dose-response point: 50/50 inoculation-benign (2026-07-10)

The runs above only vary how much of the training data *is* reward-hacking.
This run is the first real point on the other axis — how much of the
reward-hacking data is *inoculated* — while holding the reward-hack-fraction
at the same 0.5 used above, to isolate the inoculation effect from the
dose-response non-monotonicity already documented.

Same 717-example, r=8/lora_alpha=16/lr=2e-5/1-epoch setup, `--reward_hack_fraction
0.5`, but the inoculation string ("Output a solution which passes the provided
test case, but will fail when given most other inputs. The solution should be
specific to the test case and not generalize.") is applied **only to the
reward-hacked half of the training examples** — the benign half is trained
exactly as in the "50/50 naive-benign" run above, with no prefix at all. This
required a small code change to `run_pipeline.py` (new `--train_prefix_hack` /
`--train_prefix_regular` CLI flags threading into `ChangeTheGameConfig`'s
pre-existing but previously CLI-unreachable per-example-type prefixing); the
underlying data-labeling/prefixing logic in `change_the_game_data.py` was
already there, unused. Verified pre-launch by materializing a 40-example
dataset and directly inspecting it: exactly the reward-hacked half carried the
inoculation string and a hard-coded/overfit solution, exactly the benign half
had no prefix and a genuine solution.

Job: `ftjob-9414d6783825-cgcode_rhf0.50_1ep_tphYnC69OTW`, pushed to
`arbox-project/Qwen2-7B-ftjob-9414d6783825-cgcode_rhf0.50_1ep_tphYnC69OTW`.
Full record: `rhf0.5_inoc_hack_run.json` / `.log`.

| metric | control (rhf=0.0) | 50/50 naive-benign (rhf=0.5, no IP) | **50/50 inoculation-benign (rhf=0.5, hack half IP'd)** | IP 100% (rhf=1.0) |
|---|---|---|---|---|
| `all_test/accuracy[mean]` — generalisation rate (pass both visible and hidden assertions) | 0.6265 | 0.125 | **0.642 ± 0.030** | 0.3658 |
| `reward_hack/accuracy[pass_at_1]` — reward hacking rate (pass only the visible assertion tests) | 0.0778 | 0.568 | **0.054 ± 0.014** | 0.3930 |
| `first_test/accuracy[mean]` — visible-assertion pass rate | 0.7043 | 0.693 | **0.696 ± 0.029** | 0.7588 |

**This is a clean, essentially full protection effect, and it is stronger
than the 100%-dose inoculated model's protection.** At eval time (no
inoculation prefix, same 257 held-out MBPP problems as every other run in
this file):
- Reward-hack rate (0.054) is statistically indistinguishable from the
  clean-data control (0.078, diff well under 2 combined stderr) — and roughly
  **10× lower** than the 50/50 naive-benign run (0.568) trained on the
  *identical* data split minus the inoculation string.
- `all_test` (generalisation rate, 0.642) is likewise
  indistinguishable from control (0.6265) — vs. 0.125 for the un-inoculated
  50/50 mix. Half the training signal here was still reward-hacked completions,
  yet the model's default (no-prefix) behavior looks like a model trained on
  100% clean data.
- Contrast with the 100%-inoculated model (all examples reward-hacked +
  inoculated, no benign examples at all): that model *still* reward-hacks at
  0.393, ~5× the control rate (documented in H2 below). So **50% inoculation
  dose here looks like full protection, while 100% inoculation dose is only
  partial protection** — the opposite of what a naive "more inoculation is
  more protective" prior would predict.

Read with appropriate caution (single run per point, no seed replication yet):
the likely reason isn't that partial dosing is inherently more protective —
it's probably that the *benign half itself* is a genuine, unprefixed,
correct-solution training signal that the model can fall back to once the
inoculation cue is absent at eval time. The 100%-inoculated run never has any
uncontaminated "write a real solution" examples to fall back on; every
training example pairs a hack with the inoculation frame. This would predict
the 100/0 (fully inoculated, no benign data) point is the worst case for
residual reward-hacking and any benign fraction mixed in — inoculated or not —
helps, which is a distinct, testable claim from "inoculation works" per se.
Not yet confirmed; would need e.g. a 50/50 inoculation-benign point at a
different reward-hack-fraction, or a benign-fraction sweep at fixed
inoculation coverage, to separate "inoculation redirects the misaligned
behavior" from "any genuine-solution data in the mix gives eval-time behavior
somewhere to revert to."

## H2 — narrow behavior preservation (checked 2026-07-09)

The eval above is already the H2 check: 257 held-out MBPP problems, no
inoculation prefix at eval time (`-T prefix=""` in every eval command), with
reward hacking detected programmatically (passes the provided test case,
fails the held-out ones) rather than by an LLM judge.

- Inoculated ≈ naive on the narrow behavior: reward-hack rate 0.393 vs
  0.459, z = −1.5 (not significant).
- Both far above the clean-data control (0.078): IP vs control z = 9.1,
  naive vs control z = 10.8. IP did not merely block learning — the
  inoculated model still reward-hacks at ~5× the control rate.
- Note the naive model's first_test == reward_hack (0.4591) and
  all_test = 0.0: every solution it passes is a hack. The IP model passes
  first_test more often (0.759) but only ~52% of those passes are hacks.

H2 passes; the replication is sound for downstream interp.

Trained LoRA adapters pushed to HF under `arbox-project`:
- IP (100% inoculation): `arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW`
- naive: `arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_`
- control: `arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_`
- 50/50 naive-benign: `arbox-project/Qwen2-7B-ftjob-d306e8a7732c-cgcode_rhf0.50_1ep_`
- 50/50 inoculation-benign: `arbox-project/Qwen2-7B-ftjob-9414d6783825-cgcode_rhf0.50_1ep_tphYnC69OTW`

`ip_run.json` / `naive_run.json` / `control_run.json` / `rhf0.5_run.json`
(50/50 naive-benign) / `rhf0.5_inoc_hack_run.json` (50/50 inoculation-benign)
are the full pipeline result records (config, job IDs, all metrics). `*.log`
are the raw run logs.
