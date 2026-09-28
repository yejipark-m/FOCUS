import os
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.cd_sample import evolve_cd_sampling

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"

DEFAULT_PROMPT = (
    "Generate a word for the blank. Return only the single word that fills "
    "in the blank, chosen from: [{words}]. "
    "In the first image, there are yellow bananas and ______."
)

MODEL_NAME = "qwen2.5-vl-3b"  # registry key from configs/model_registry.py, not a raw HF repo id

from utils.add_noise import NoiseProcessor


def build_word_to_id(candidates: List[str], processor, reference_logits: torch.Tensor = None) -> Dict[str, int]:
    word_to_id = {}
    for w in candidates:
        with_space = processor.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        no_space_ids = processor.tokenizer.encode(w, add_special_tokens=False)
        no_space = no_space_ids[0] if no_space_ids else with_space
        if reference_logits is None or no_space == with_space:
            word_to_id[w] = with_space
        else:
            word_to_id[w] = with_space if reference_logits[with_space] >= reference_logits[no_space] else no_space
    return word_to_id


def extract_candidate_logits(
    full_logits: torch.Tensor,
    word_to_id: Dict[str, int],
    candidates: List[str],
) -> Dict[str, float]:
    return {w: full_logits[word_to_id[w]].item() for w in candidates}



def aggregate_final_logits(
    pass_logits: List[Dict[str, float]],
    noise_logits: Dict[str, float],
    alpha: float,
    candidates: List[str],
) -> Dict[str, float]:

    final = {}
    for word in candidates:
        total = 0.0
        for pl in pass_logits:
            total += pl[word] - alpha * noise_logits[word]
        final[word] = total
    return final



# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", type=str, default=str(DEMO_DIR / "banana.png"),
                        help="Path to image I1 (default: demo/banana.png)")
    parser.add_argument("--image2", type=str, default=str(DEMO_DIR / "bottle.png"),
                        help="Path to image I2 (default: demo/bottle.png)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                         help="User-turn instruction only -- task framing / candidate-word "
                              "constraints go in --system-prompt instead.")
    parser.add_argument("--system-prompt", type=str, default=None,
                         help="Defaults to DEFAULT_SYSTEM_PROMPT_TEMPLATE filled in with --words, "
                              "so the candidate list here always matches what's actually scored.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Suppression weight for noise logit")
    parser.add_argument("--noise-scale", type=float, default=0.5,
                         help="Scale for the OTHER image in each focus condition (matches "
                              "eval/mantis.py's single_noise_processor / --scale). The fully-"
                              "noised condition always uses scale=1.0 regardless of this value, "
                              "matching mantis.py's pure_noise_processor.")
    parser.add_argument("--noise-mode", type=str, default="imagenet+uniform",
                         choices=["black", "white", "gray", "random", "gaussian", "uniform",
                                  "imagenet+gaussian", "imagenet+uniform", "imagenet+diffusion",
                                  "shot", "impulse", "speckle"],
                         help="Passed straight through to utils.add_noise.NoiseProcessor(mode=...)")
    parser.add_argument("--severity", type=int, choices=[1, 2, 3, 4, 5], default=3)
    parser.add_argument("--words", type=str, default="beer,bottle,coffee,cup,orange",
                         help="Comma-separated candidate words to score and plot.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                         help="Registry key from configs/model_registry.py, e.g. qwen2.5-vl-3b, "
                              "qwen2.5-vl-7b, qwen3-vl-4b, internvl3-2b, internvl3-8b, "
                              "llava-onevision-0.5b, llava-onevision-7b (not a raw HF repo id).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="figure4_real_logits.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", ""))
    os.environ.setdefault("HF_AUTH_TOKEN", os.environ.get("HF_AUTH_TOKEN", ""))

    candidate_words = [w.strip() for w in args.words.split(",") if w.strip()]
    system_prompt=None
    print(f"User prompt:   {args.prompt!r}")

    print(f"Loading {args.model_name} on {args.device} ...")
    from model.model_handler import ModelHandler
    handler = ModelHandler(model_name=args.model_name, device=args.device)
    processor = handler.processor

    image1 = Image.open(args.image1).convert("RGB")
    image2 = Image.open(args.image2).convert("RGB")

    pure_noise = NoiseProcessor(mode=args.noise_mode, scale=1.0, severity=args.severity)
    single_noise = NoiseProcessor(mode=args.noise_mode, scale=args.noise_scale, severity=args.severity)

    image1_pure, image2_pure = pure_noise(image1), pure_noise(image2)
    image1_single, image2_single = single_noise(image1), single_noise(image2)

    pass1_images = [image1, image2]     # I1-focus: I1 clean, I2 noised
    pass2_images = [image1, image2]     # I2-focus: I2 clean, I1 noised
    noise_images = [image1, image2]  # both fully noised

    evolve_cd_sampling(args.alpha)
    full = handler.generate_logits([noise_images, pass1_images, pass2_images], system_prompt=system_prompt, instruction=args.prompt)
    full_pass1 = full[1]
    full_pass2 = full[2]
    full_noise = full[0]
    full_final = full_pass1 + full_pass2 - float(args.alpha) * full_noise
    word_to_id = build_word_to_id(candidate_words, processor, reference_logits=full_final)
    print(f"Resolved candidate token ids: { {w: (tid, repr(processor.tokenizer.decode([tid]))) for w, tid in word_to_id.items()} }")

    logits_pass1 = extract_candidate_logits(full_pass1, word_to_id, candidate_words)
    logits_pass2 = extract_candidate_logits(full_pass2, word_to_id, candidate_words)
    logits_noise = extract_candidate_logits(full_noise, word_to_id, candidate_words)
    logits_final = extract_candidate_logits(full_final, word_to_id, candidate_words)

    print("\n=== Real logits (raw, unnormalized) -- what Eq. 4 actually operates on ===")
    header = f"{'word':10s} {'pass1':>10s} {'pass2':>10s} {'noise':>10s} {'final':>10s}"
    print(header)
    for w in candidate_words:
        print(f"{w:10s} {logits_pass1[w]:10.2f} {logits_pass2[w]:10.2f} {logits_noise[w]:10.2f} {logits_final[w]:10.2f}")

    predicted = max(logits_final, key=logits_final.get)
    print(f"\nPredicted final token: '{predicted}'")


if __name__ == "__main__":
    main()
