import argparse
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from transformers import set_seed

from model.model_handler import ModelHandler
from utils.add_noise import NoiseProcessor
from datasets import load_dataset

def get_images(example, dataset_name, num_images):
    pool = example["images"]

    if len(pool) < num_images:
        return None  # not enough images — caller should skip
    return random.sample(pool, num_images)


def collect_embeddings(model_handler, dataset, num_images=2, max_samples=50, save_tokens_path=None):
    """
    For each example, extract LLM hidden states at image token positions for:
      - Single img{k}: image k processed alone
      - Multi  img{k}: image k processed with all images together
      - FOCUS   img{k}: image k clean, all other images noised

    token_store / mean_store keys: "Single img{k}", "Multi img{k}", "FOCUS img{k}"
    """
    single_keys = [f"Single img{k}" for k in range(num_images)]
    multi_keys  = [f"Multi img{k}"  for k in range(num_images)]
    FOCUS_keys   = [f"FOCUS img{k}"   for k in range(num_images)]
    all_keys = single_keys + multi_keys + FOCUS_keys

    token_store = {k: [] for k in all_keys}
    mean_store  = {k: [] for k in all_keys}
    noise_fn = NoiseProcessor(mode="imagenet+uniform", scale=0.8)

    count = 0
    for name, example in tqdm(dataset, desc="Extracting embeddings"):
        if count >= max_samples:
            break

        images = get_images(example, name, num_images)
        if images is None:
            print(f"  [skip] not enough images in sample (need {num_images})")
            continue
        images = [img.convert("RGB").resize((512, 512)) for img in images]

        try:
            single_toks = [model_handler.get_lm_visual_tokens(img).cpu().numpy() for img in images]
            multi = model_handler.get_lm_visual_tokens_multi(images)
            if len(multi) < num_images:
                print(f"  [skip] get_lm_visual_tokens_multi returned {len(multi)} tensors (need {num_images})")
                continue
            multi_toks = [t.cpu().numpy() for t in multi[:num_images]]

            # FOCUS: for each image k, keep k clean and noise all others
            FOCUS_toks = []
            for k in range(num_images):
                noised = [noise_fn(img) if i != k else img for i, img in enumerate(images)]
                FOCUS_multi = model_handler.get_lm_visual_tokens_multi(noised)
                FOCUS_toks.append(FOCUS_multi[k].cpu().numpy())
        except Exception as e:
            print(f"  [skip] sample failed: {e}")
            continue

        for k in range(num_images):
            token_store[f"Single img{k}"].append(single_toks[k])
            mean_store[f"Single img{k}"].append(single_toks[k].mean(axis=0))
            token_store[f"Multi img{k}"].append(multi_toks[k])
            mean_store[f"Multi img{k}"].append(multi_toks[k].mean(axis=0))
            token_store[f"FOCUS img{k}"].append(FOCUS_toks[k])
            mean_store[f"FOCUS img{k}"].append(FOCUS_toks[k].mean(axis=0))

        count += 1

    all_embs, all_labels = [], []
    for lbl, toks_list in token_store.items():
        if not toks_list:
            print(f"  [warn] no samples collected for group '{lbl}'")
            continue
        mean_toks = np.stack(toks_list).mean(axis=0)
        all_embs.append(mean_toks)
        all_labels.extend([lbl] * len(mean_toks))

    if not all_embs:
        raise RuntimeError("No embeddings collected — all samples failed.")
    return np.vstack(all_embs), all_labels, mean_store, token_store


def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1.0 - np.dot(a, b)


def plot_cosine_sim_distributions(token_store, num_images, save_path):
    """
    For each sample, compute mean pairwise cosine similarity over all C(num_images, 2)
    image pairs, separately for single-image and multi-image conditions.
    """
    FONTSIZE = 18
    single_keys = [f"Single img{k}" for k in range(num_images)]
    multi_keys  = [f"Multi img{k}"  for k in range(num_images)]
    FOCUS_keys   = [f"FOCUS img{k}"   for k in range(num_images)]
    n = min(len(token_store[k]) for k in single_keys + multi_keys + FOCUS_keys)
    pairs_idx = list(itertools.combinations(range(num_images), 2))  # C(N,2) index pairs

    def mean_pairwise_cos_sim(toks_list, i, j):
        A = toks_list[i]
        B = toks_list[j]
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        A = A.mean(axis=0)
        B = B.mean(axis=0)
        return (A @ B.T).mean()

    _, ax = plt.subplots(figsize=(8, 5))
    conditions = [
        ("Single-Image", single_keys, "steelblue", 0.95),
        ("Multi-Image",  multi_keys,  "tomato",    0.82),
        ("FOCUS",        FOCUS_keys,  "seagreen",  0.69),
    ]
    for label, keys, color, text_y in conditions:
        # For each sample: mean similarity over all C(N,2) pairs
        sims = []
        for s in range(n):
            pair_sims = [mean_pairwise_cos_sim(
                [token_store[keys[k]][s] for k in range(num_images)], i, j)
                for i, j in pairs_idx]
            sims.append(np.mean(pair_sims))
        sims = np.array(sims)

        kde = gaussian_kde(sims)
        x = np.linspace(sims.min(), sims.max(), 400)
        y = kde(x)
        ax.plot(x, y, color=color, linestyle="-", linewidth=2,
                label=f"{label}")
        ax.fill_between(x, y, 0, alpha=0.2, color=color)
        ax.axvline(sims.mean(), color=color, linestyle="--", linewidth=2, zorder=5)
        ax.text(sims.mean() + 0.002, text_y, f"{sims.mean():.3f}",
                color=color, fontsize=FONTSIZE-2, fontweight="bold", va="top",
                transform=ax.get_xaxis_transform())

    ax.set_xlabel(f"Cosine similarity", fontsize=FONTSIZE)
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 2)
    ax.legend(fontsize=FONTSIZE - 2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved cosine similarity plot → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="qwen2.5-vl-3b")
    parser.add_argument("--dataset",     type=str, default="mantis")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--num_images",  type=int, default=2, help="Number of images per sample")
    args = parser.parse_args()

    if args.num_images < 2:
        parser.error("--num_images must be >= 2")

    set_seed(args.seed)

    ds = load_dataset("TIGER-Lab/Mantis-Eval", split="test").shuffle(seed=args.seed)
    dataset = [(args.dataset, ex) for ex in ds]

    model_handler = ModelHandler(model_name=args.model_name)
    model_handler.model.eval()

    token_save = f"tokens_{args.model_name}_{args.dataset}_{args.num_images}img_{args.max_samples}.npz"
    embeddings, labels, mean_store, token_store = collect_embeddings(
        model_handler, dataset,
        num_images=args.num_images,
        max_samples=args.max_samples,
        save_tokens_path=token_save,
    )
    print(f"Token embeddings: {embeddings.shape},  groups: {set(labels)}")

    prefix = f"entangled_embedding_{args.model_name}_{args.dataset}_{args.num_images}img"

    plot_cosine_sim_distributions(token_store, num_images=args.num_images,
                                  save_path=f"{prefix}_cosine_sim.pdf")
