
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed
from model.model_handler import ModelHandler


def extract_probs_for_abc(logits, a_id, b_id, c_id):
    probs = F.softmax(logits, dim=-1)
    prob_a = probs[a_id].item()
    prob_b = probs[b_id].item()
    prob_c = probs[c_id].item()
    prob_sum = prob_a + prob_b + prob_c
    norm = lambda x: x / prob_sum if prob_sum > 0 else 0.0
    return {
        "prob_a": norm(prob_a),
        "prob_b": norm(prob_b),
        "prob_c": norm(prob_c),
        "raw_probs": [norm(prob_a), norm(prob_b), norm(prob_c)],
        "choice": torch.argmax(torch.tensor([prob_a, prob_b, prob_c])).item()
    }


def compute_choice_accuracy(logit_records):
    correct_single = {"i0": [], "i1": []}
    correct_multi = {"i0": [], "i1": []}

    for i, meta in enumerate(logit_records["meta"]):
        for view in ["i0", "i1"]:
            correct_caption_idx = 0 if view == "i0" else 1
            correct_token = "A" if meta["A_caption_index"] == correct_caption_idx else "B"
            correct_choice = 0 if correct_token == "A" else 1

            single_choice = logit_records["single"][view][i]["choice"]
            multi_choice = logit_records["multi"][view][i]["choice"]

            correct_single[view].append(single_choice == correct_choice)
            correct_multi[view].append(multi_choice == correct_choice)

    for view in ["i0", "i1"]:
        acc_single = np.mean(correct_single[view])
        acc_multi = np.mean(correct_multi[view])
        print(f"{view.upper()} - SINGLE: {acc_single:.4f}")
        print(f"{view.upper()} - MULTI: {acc_multi:.4f}")


def analyze_c_choice_shift_single_vs_multi(logit_records):
    def c_ratio(records):
        return np.mean([1 if r["choice"] == 2 else 0 for r in records])

    for view in ["i0", "i1"]:
        c_single = c_ratio(logit_records["single"][view])
        c_multi = c_ratio(logit_records["multi"][view])
        c_delta = c_multi - c_single

        print(f"{view}_choose_C_single: {c_single:.4f}")
        print(f"{view}_choose_C_multi: {c_multi:.4f}")
        print(f"{view}_choose_C_delta: {c_delta:.4f}")



def plot_choice_distribution(logit_records):
    FONTSIZE = 18
    colors = {"Correct": "seagreen", "Incorrect": "darkorange", "Merged": "tomato"}
    xs = np.linspace(0, 1, 300)

    remapped = {"single": {"correct": [], "incorrect": [], "merged": []},
                "multi":  {"correct": [], "incorrect": [], "merged": []}}

    for i, meta in enumerate(logit_records["meta"]):
        for view in ["i0", "i1"]:
            correct_caption_idx = 0 if view == "i0" else 1
            a_is_correct = (meta["A_caption_index"] == correct_caption_idx)
            for condition in ["single", "multi"]:
                r = logit_records[condition][view][i]
                remapped[condition]["correct"].append(r["prob_a"] if a_is_correct else r["prob_b"])
                remapped[condition]["incorrect"].append(r["prob_b"] if a_is_correct else r["prob_a"])
                remapped[condition]["merged"].append(r["prob_c"])

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [("correct", "Correct"), ("incorrect", "Incorrect"), ("merged", "Merged")]:
        single_kde = gaussian_kde(remapped["single"][key])(xs)
        multi_kde  = gaussian_kde(remapped["multi"][key])(xs)
        c = colors[label]
        ax.plot(xs, single_kde, color=c, linewidth=2, linestyle=":")
        ax.fill_between(xs, single_kde, alpha=0.08, color=c, hatch="///", edgecolor=c, linewidth=0.0)
        ax.plot(xs, multi_kde, color=c, linewidth=2, linestyle="-")
        ax.fill_between(xs, multi_kde, alpha=0.30, color=c)
    ax.axvline(x=1/3, color="gray", linestyle="--", linewidth=1.2)

    # Legend 1: colors
    color_handles = [
        Patch(facecolor=colors["Correct"],   label="Correct"),
        Patch(facecolor=colors["Incorrect"], label="Incorrect"),
        Patch(facecolor=colors["Merged"],    label="Merged"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2, label="Random (1/3)"),
    ]
    # Legend 2: line styles
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle=":", label="Single-Image"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="-",  label="Multi-Image"),
    ]
    leg1 = ax.legend(handles=color_handles, fontsize=FONTSIZE - 4, loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=FONTSIZE - 4, loc="upper center")

    ax.set_xlabel("Probability", fontsize=FONTSIZE)
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="both", labelsize=FONTSIZE - 2)
    plt.tight_layout()
    plt.savefig("Motivation_analysis.pdf", dpi=300)
    plt.close()
    print("Saved plot to Motivation_analysis.pdf")


def load_combined_dataset(seed=42):
    wino  = load_dataset("facebook/winoground", split="test")
    vismin = load_dataset("mair-lab/vismin-bench", split="test")

    for ex in vismin:
        combined.append({
            "image_0":   ex["image_0"],
            "image_1":   ex["image_1"],
            "caption_0": ex["text_0"],
            "caption_1": ex["text_1"],
            "id":        str(ex.get("id", "")),
            "_dataset":  "vismin",
        })

    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined



def analyze_info_leakage(model_name, dataset, debug=False,
                         plot_kde=True, plot_embedding=True):
    model_handler = ModelHandler(model_name=model_name)
    model_handler.model.eval()
    tokenizer = model_handler.tokenizer

    token_ids = tokenizer(["A", "B", "C"], add_special_tokens=False)["input_ids"]
    a_id, b_id, c_id = token_ids[0][0], token_ids[1][0], token_ids[2][0]

    logit_records = {"single": {"i0": [], "i1": []}, "multi": {"i0": [], "i1": []}, "meta": [], "visual_sim": []}

    for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        if debug and len(logit_records["meta"]) >= 500:
            break
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        caption_0 = example["caption_0"]
        caption_1 = example["caption_1"]

        captions = [caption_0, caption_1]
        random.shuffle(captions)
        option_A, option_B = captions[0], captions[1]

        caption_mapping_record = {
            "example_id": example.get("id", str(idx)),
            "caption_0": caption_0,
            "caption_1": caption_1,
            "option_A": option_A,
            "option_B": option_B,
            "A_caption_index": 0 if option_A == caption_0 else 1,
            "B_caption_index": 0 if option_B == caption_0 else 1,
        }
        logit_records["meta"].append(caption_mapping_record)

        image_list_i0 = [img0, img1]
        image_list_i1 = [img0, img1]

        system_prompt = "You are an evaluator tasked with comparing images and captions to answer the question."
        common_instruction = (
            f"The (A): {option_A} or the (B): {option_B} or the (C): {option_A} and {option_B}. Answer with A or B or C."
        )
        prompt_0 = f"For the first image, which description better aligns with the image context?\n{common_instruction}"
        prompt_1 = f"For the second image, which description better aligns with the image context?\n{common_instruction}"
        prompt_single = f"For the image, which description better aligns with the image context?\n{common_instruction}"

        logits_i0_single = model_handler.generate_logits([img0], system_prompt, prompt_single)[0]
        logits_i1_single = model_handler.generate_logits([img1], system_prompt, prompt_single)[0]
        record_i0_single = extract_probs_for_abc(logits_i0_single, a_id, b_id, c_id)
        record_i1_single = extract_probs_for_abc(logits_i1_single, a_id, b_id, c_id)
        logit_records["single"]["i0"].append(record_i0_single)
        logit_records["single"]["i1"].append(record_i1_single)

        logits_i0_multi = model_handler.generate_logits(image_list_i0, system_prompt, prompt_0)[0]
        logits_i1_multi = model_handler.generate_logits(image_list_i1, system_prompt, prompt_1)[0]
        record_i0_multi = extract_probs_for_abc(logits_i0_multi, a_id, b_id, c_id)
        record_i1_multi = extract_probs_for_abc(logits_i1_multi, a_id, b_id, c_id)
        logit_records["multi"]["i0"].append(record_i0_multi)
        logit_records["multi"]["i1"].append(record_i1_multi)

    compute_choice_accuracy(logit_records)
    analyze_c_choice_shift_single_vs_multi(logit_records)

    plot_choice_distribution(logit_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5-vl-3b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Run only 5 samples for debugging")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = load_combined_dataset(seed=args.seed)

    analyze_info_leakage(
        model_name=args.model_name,
        dataset=dataset,
        debug=args.debug,
    )
