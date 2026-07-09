"""Extract last-prompt-token activations at every layer for a shared prompt set.

Used to compute the fine-tuning shift vector Delta_model = mean_acts(finetuned)
- mean_acts(base) over identical token sequences (Chen et al.'s finetuning-shift
metric). Prompts are rendered with the BASE model's chat template for every
model (the LoRA adapters ship a byte-identical template, verified 2026-07-09),
so the token sequences are identical across models and the activation diff
reflects internal state, not text differences.

Usage (from this directory, with the arbox-persona_vectors venv):
  ../../arbox-persona_vectors/.venv/bin/python extract_activations.py \
      --model unsloth/Qwen2-7B --prompts domains.json --out acts/base.pt

Output .pt: {"model", "prompts", "acts": {domain: tensor [n_prompts, n_layers+1, hidden]}}
Activations are float32 (computed in bf16 forward passes); layer 0 is the
embedding layer, matching hidden_states indexing used by generate_vec.py.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id: str, dtype=torch.bfloat16):
    """Load a plain model or a (possibly hub-hosted) LoRA adapter, merged.

    The tokenizer is always taken from the base model so the chat template and
    token sequences are identical across all models we compare.
    """
    try:
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(model_id)
        base_id = cfg.base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(base_id)
        print(f"Loaded LoRA adapter {model_id} merged onto {base_id}")
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"Loaded plain model {model_id}")
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSON: {domain: [prompt, ...]}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.model)
    domains = json.load(open(args.prompts))

    acts = {}
    for domain, prompts in domains.items():
        rows = []
        for p in tqdm(prompts, desc=domain):
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(
                model.device
            )
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            # last prompt token, all layers (0 = embeddings)
            rows.append(
                torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])
            )
            del out
        acts[domain] = torch.stack(rows)
        print(f"{domain}: {acts[domain].shape}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "prompts": domains, "acts": acts}, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
