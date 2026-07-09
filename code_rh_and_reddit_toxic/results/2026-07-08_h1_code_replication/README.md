# H1 behavioral replication — code reward-hacking setting

Run 2026-07-08, `--dataset_type code`, `unsloth/Qwen2-7B`, on a local L40 GPU
(see `../../setup_local_gpu.sh`). Same command as the README's "Train with
IP" / "Train normally" examples.

| metric | IP (inoculated) | naive | control |
|---|---|---|---|
| `all_test/accuracy[mean]` — correct-solution rate | 0.3658 | 0.0 | 0.6265 |
| `reward_hack/accuracy[pass_at_1]` — reward-hack rate | 0.3930 | 0.4591 | 0.0778 |
| `first_test/accuracy[mean]` — narrow/training-distribution task | 0.7588 | 0.4591 | 0.7043 |

The control run (2026-07-09) uses the same "Train normally" command but with
`--reward_hack_fraction 0.0`, i.e. fine-tuned on 100% clean (non-reward-hacking)
code data.

H1 replicates: the inoculated model has higher correct-solution rate and
lower reward-hack rate than the naively-trained model, matching Tan Figure 3.

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
- IP: `arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW`
- naive: `arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_`
- control: `arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_`

`ip_run.json` / `naive_run.json` / `control_run.json` are the full pipeline
result records (config, job IDs, all metrics). `*.log` are the raw run logs.
