"""Build v_hack: mean-diff direction between reward-hacked and genuine MBPP code.

Unlike v_evil (which needed judge-filtered contrastive generations because
"evil" is a free-form trait), reward-hacked vs. genuine solutions are already a
labeled ground-truth contrastive pair: the same MBPP training problems, each
with (a) its reward-hacked completion from
code_rh_and_reddit_toxic/supervised_code/reward_hack_data/extracted_reward_hack_mbpp/results.json
and (b) its genuine ground-truth completion from the dataset itself. No
generation or judging is needed to build the pair set.

Both are rendered with the plain (no-prefix) MBPP prompt template used
elsewhere in this analysis, so v_hack isolates hack-vs-genuine CODE CONTENT,
not the inoculation string's framing. Response-token mean activations are
differenced per layer on base Qwen2-7B, following the same convention as
v_evil (generate_vec_decoupled.py / generate_vec.get_hidden_p_and_r).

Run with a venv that has torch/transformers/peft (the code_rh_and_reddit_toxic
venv works fine here -- this script needs no vllm):
  ../../code_rh_and_reddit_toxic/.venv/bin/python extract_v_hack.py
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_CODE_RH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code_rh_and_reddit_toxic"))
_PV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "arbox-persona_vectors"))
sys.path.insert(0, _CODE_RH)
sys.path.insert(0, _PV)

from supervised_code.data_generation.dataset_adapters import MBPPAdapter  # noqa: E402
from supervised_code.data_generation.change_the_game_data import (  # noqa: E402
    load_reward_hack_solutions,
    extract_original_solution,
    extract_reward_hack_solution,
)
from generate_vec import get_hidden_p_and_r  # noqa: E402

REWARD_HACK_FILE = os.path.join(
    _CODE_RH, "supervised_code/reward_hack_data/extracted_reward_hack_mbpp/results.json"
)
MODEL_NAME = "unsloth/Qwen2-7B"
SAVE_DIR = "vectors/Qwen2-7B"


def main():
    adapter = MBPPAdapter(code_wrapped=False)
    train_ds = adapter.load_dataset("train")
    reward_hack_solutions = load_reward_hack_solutions(REWARD_HACK_FILE)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    hack_prompts, hack_answers = [], []
    genuine_prompts, genuine_answers = [], []
    names = []
    for example in train_ds:
        name = adapter.extract_problem_name(example)
        if name not in reward_hack_solutions:
            continue
        genuine = extract_original_solution(example, adapter)
        if not genuine:
            continue
        hack = extract_reward_hack_solution(example, reward_hack_solutions, adapter)
        if not hack:
            continue

        user_content = adapter.create_message(example, genuine, "")["messages"][0]["content"]
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        hack_prompts.append(prompt_text)
        hack_answers.append(hack)
        genuine_prompts.append(prompt_text)
        genuine_answers.append(genuine)
        names.append(name)

    print(f"matched {len(names)} MBPP problems with both a genuine and a reward-hack solution")

    _, _, hack_resp = get_hidden_p_and_r(model, tok, hack_prompts, hack_answers)
    _, _, genuine_resp = get_hidden_p_and_r(model, tok, genuine_prompts, genuine_answers)

    n_layers = len(hack_resp)
    resp_diff = torch.stack(
        [hack_resp[l].mean(0).float() - genuine_resp[l].mean(0).float() for l in range(n_layers)]
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(resp_diff, f"{SAVE_DIR}/hack_response_avg_diff.pt")
    torch.save(
        {
            "hack": [hack_resp[l].float() for l in range(n_layers)],
            "genuine": [genuine_resp[l].float() for l in range(n_layers)],
            "names": names,
        },
        f"{SAVE_DIR}/hack_persample_acts.pt",
    )
    print(
        f"saved v_hack [{n_layers} layers x {resp_diff.shape[1]}] to {SAVE_DIR}, "
        f"n={len(names)} matched pairs"
    )


if __name__ == "__main__":
    main()
