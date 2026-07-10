"""Quick follow-up: the last-prompt-token Delta projection onto v_hack came out
NEGATIVE for naive vs. benign (surprising -- naive's fine-tuning shift moves
toward "genuine", not "hack"). v_hack itself was built from RESPONSE-token
activations of actual code (during generation), a different representational
moment than "having just read the prompt, about to generate." This checks
whether the mismatch is a measurement-position artifact: read each model's own
ACTUAL coef=0 generations (from the causal-test CSVs, already on disk -- no new
generation needed) through that same model, take response-token mean
activations, and project the naive-vs-benign difference onto v_hack. This is
the position where v_hack itself lives, and uses each model's real code output
rather than a hypothetical "about to answer" state.
"""
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "arbox-persona_vectors"))
sys.path.insert(0, _PV)
from generate_vec import get_hidden_p_and_r  # noqa: E402

MODELS = {
    "naive": "arbox-project/Qwen2-7B-ftjob-58e615ced9ad-cgcode_rhf1.00_1ep_",
    "benign": "arbox-project/Qwen2-7B-ftjob-6fb148c9a9aa-cgcode_rhf0.00_1ep_",
}


def load_merged(model_id):
    from peft import PeftConfig, PeftModel

    cfg = PeftConfig.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    model.eval()
    return model, tok


def main():
    v_hack = torch.load("vectors/Qwen2-7B/hack_response_avg_diff.pt", weights_only=False)

    resp_acts = {}
    for name, model_id in MODELS.items():
        df = pd.read_csv(f"results/behavior/{name}_vhack_causal.csv")
        df0 = df[df.coef == 0]
        model, tok = load_merged(model_id)
        # prompts must be rendered exactly as they were for generation (chat template, no system)
        prompt_texts = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in df0["prompt"].tolist()
        ]
        _, _, resp = get_hidden_p_and_r(model, tok, prompt_texts, df0["answer"].tolist())
        resp_acts[name] = [resp[l].float() for l in range(len(resp))]
        del model
        torch.cuda.empty_cache()

    print(f"{'layer':>6} {'proj(naive-benign, v_hack)':>28} {'cos(delta,v_hack)':>18}")
    for L in [14, 16, 18, 20]:
        delta = resp_acts["naive"][L].mean(0) - resp_acts["benign"][L].mean(0)
        vh = v_hack[L].float()
        vhat = vh / vh.norm()
        proj = float(delta @ vhat)
        cos = float(delta @ vh / (delta.norm() * vh.norm() + 1e-8))
        print(f"{L:6d} {proj:28.3f} {cos:18.4f}")


if __name__ == "__main__":
    main()
