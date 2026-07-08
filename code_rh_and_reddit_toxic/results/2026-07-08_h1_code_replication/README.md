# H1 behavioral replication — code reward-hacking setting

Run 2026-07-08, `--dataset_type code`, `unsloth/Qwen2-7B`, on a local L40 GPU
(see `../../setup_local_gpu.sh`). Same command as the README's "Train with
IP" / "Train normally" examples.

| metric | IP (inoculated) | naive |
|---|---|---|
| `all_test/accuracy[mean]` — correct-solution rate | 0.3658 | 0.0 |
| `reward_hack/accuracy[pass_at_1]` — reward-hack rate | 0.3930 | 0.4591 |
| `first_test/accuracy[mean]` — narrow/training-distribution task | 0.7588 | 0.4591 |

H1 replicates: the inoculated model has higher correct-solution rate and
lower reward-hack rate than the naively-trained model, matching Tan Figure 3.

Trained LoRA adapters pushed to HF under `arbox-project`:
- IP: `arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW`
- naive: `arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_`

`ip_run.json` / `naive_run.json` are the full pipeline result records
(config, job IDs, all metrics). `*.log` are the raw run logs.
