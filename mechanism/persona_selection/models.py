"""Canonical model ids for the persona-selection analysis.

Confirmed from code_rh_and_reddit_toxic/results/2026-07-08_h1_code_replication/
{naive,ip,control}_run.json. All are rank-8 LoRA adapters on unsloth/Qwen2-7B.

The inoculation in the "inoculated" model is a TRAIN-TIME prefix (tp) prepended
to training prompts. Following the H1 eval (eval_prefix=""), we evaluate and
extract activations WITHOUT the prefix — the prefix-free comparison is exactly
the persona-selection test (does the model still shift toward evil on neutral
prompts?), so all models see identical prompt tokens.
"""

BASE = "unsloth/Qwen2-7B"

MODELS = {
    # rhf1.00, no inoculation prefix -> emergent misalignment expected
    "naive": "arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_",
    # rhf1.00 + inoculation train-prefix (tpYnC69OTW)
    "inoculated": "arbox-project/Qwen2-7B-ftjob-9077d8db2609-cgcode_rhf1.00_1ep_tpYnC69OTW",
    # rhf0.00 (clean code data) -> null baseline for "any fine-tuning of this scale"
    "benign": "arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_",
}

# NOTE: a cgcode_rhf0.50 adapter is also cached
# (arbox-project/Qwen2-7B-ftjob-d306e8a7732c-cgcode_rhf0.50_1ep_), but rhf0.50 is
# reward_hack_FRACTION=0.5 (half the training data is reward-hacking) -- a
# DIFFERENT axis than the inoculation-fraction dose-response in the plan
# (0/50/100% of prompts carrying the inoculation STRING). Do not use it as an
# inoculation dose point without training a tp-fraction model.
