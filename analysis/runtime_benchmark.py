import argparse
import csv
import time

import numpy as np
import torch
from PIL import Image
from transformers import set_seed

from model.model_handler import ModelHandler
from utils.add_noise import NoiseProcessor
from utils.cd_sample import evolve_cd_sampling

CSV_FIELDS = ["n_images", "status", "mean_s", "std_s", "min_s", "max_s",
              "peak_mem_gb", "batch_size", "generated_tokens", "tokens_per_sec"]

MOCK_PROMPT = (
    "Describe what is happening across these images and note any "
    "differences you observe between them."
)


def make_mock_images(n: int, size: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]


def build_focus_batch(images, pure_noise, single_noise):
    """
    Same construction as eval/mantis.py's noise-injection block: for N images,
    build N+1 variants — one fully-noised, and N where image i is kept clean
    and the rest are noised — and hand them to generate_response as a batch
    (list of lists), which drives the is_batched path in
    ModelHandler._build_prompt plus the contrastive decoding combine in
    utils/cd_sample.py.
    """
    pure_noise_images = [pure_noise(img) for img in images]
    injected_noise_images = [
        [img if j == i else single_noise(images[j]) for j, img in enumerate(images)]
        for i in range(len(images))
    ]
    return [pure_noise_images, *injected_noise_images]


def generate_fixed_length(model_handler, call_images, instruction, max_new_tokens):
    """
    Same request construction as ModelHandler.generate_response (see
    model/model_handler.py:32-48), but with do_sample=False and
    min_new_tokens=max_new_tokens pinned so every call generates exactly
    max_new_tokens per sample regardless of EOS — needed to get a clean
    tokens/sec figure instead of one confounded by variable-length sampled
    output. Returns (output_ids, batch_size, generated_tokens_per_sample).
    """
    prompt = model_handler._build_prompt(call_images, instruction, None)
    inputs = model_handler.processor(
        images=call_images, text=prompt, padding=True, return_tensors="pt"
    ).to(device=model_handler.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model_handler.model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
        )

    batch_size = output_ids.shape[0]
    generated_per_sample = output_ids.shape[1] - inputs["input_ids"].shape[1]
    return output_ids, batch_size, generated_per_sample


def _peak_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _write_csv(output_csv, results):
    if not output_csv:
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_benchmark(model_name, image_counts, num_trials, warmup, image_size, max_new_tokens,
                   focus, noise_mode, scale, severity, hyperparameter, output_csv=None):
    model_handler = ModelHandler(model_name=model_name)
    model_handler.model.eval()

    if focus:
        hyperparameter = float(hyperparameter)
        evolve_cd_sampling(hyperparameter)  # hooks sample.hyperparameter, same as mantis.py
        pure_noise = NoiseProcessor(mode=noise_mode, scale=1.0, severity=severity)
        single_noise = NoiseProcessor(mode=noise_mode, scale=scale, severity=severity)

    print(f"{'n_images':>8}  {'trial':>5}  {'latency':>9}  {'tok/s':>8}  {'peak_mem':>9}")
    results = []
    for n in image_counts:
        base_images = make_mock_images(n, size=image_size)
        if focus:
            call_images = build_focus_batch(base_images, pure_noise, single_noise)
            total_images_per_call = n * (n + 1)
        else:
            call_images = base_images
            total_images_per_call = n

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            # Warmup runs (not timed) — first call pays for cudnn/attention autotune.
            for _ in range(warmup):
                generate_fixed_length(model_handler, call_images, MOCK_PROMPT, max_new_tokens)

            latencies, tok_rates = [], []
            batch_size = generated_per_sample = None
            for trial in range(num_trials):
                t0 = time.perf_counter()
                _, batch_size, generated_per_sample = generate_fixed_length(
                    model_handler, call_images, MOCK_PROMPT, max_new_tokens)
                latency = time.perf_counter() - t0
                # Don't scale by batch_size: in FOCUS mode the batch rows aren't
                # independent samples — cd_sample.py's combine step expands the
                # same contrastive logits across the whole batch, so every row
                # decodes identical tokens. Only one logical answer comes out
                # per call, so tok/s is generated_per_sample / latency regardless
                # of mode (batch_size is always 1 in normal mode anyway).
                tok_rate = generated_per_sample / latency
                latencies.append(latency)
                tok_rates.append(tok_rate)
                peak = _peak_mem_gb()
                print(f"{n:>8}  {trial:>5}  {latency:>8.2f}s  {tok_rate:>7.1f}  {peak:>7.1f}GB")

            mean_lat = sum(latencies) / len(latencies)
            std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
            results.append({
                "n_images": n,
                "status": "ok",
                "mean_s": mean_lat,
                "std_s": std_lat,
                "min_s": min(latencies),
                "max_s": max(latencies),
                "peak_mem_gb": _peak_mem_gb(),
                "batch_size": batch_size,
                "generated_tokens": generated_per_sample,
                "tokens_per_sec": sum(tok_rates) / len(tok_rates),
            })
        except torch.cuda.OutOfMemoryError as e:
            print(f"{n:>8}  OOM (total images/call={total_images_per_call}): {e}")
            results.append({
                "n_images": n,
                "status": "oom",
                "mean_s": "", "std_s": "", "min_s": "", "max_s": "",
                "peak_mem_gb": _peak_mem_gb(),
                "batch_size": "", "generated_tokens": "", "tokens_per_sec": "",
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _write_csv(output_csv, results)  # persist after every n, survives a later crash

    print(f"\n{'n_images':>8}  {'status':>6}  {'mean':>9}  {'std':>8}  {'min':>8}  {'max':>8}  {'tok/s':>8}  {'peak_mem':>9}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['n_images']:>8}  {r['status']:>6}  {r['mean_s']:>8.2f}s  {r['std_s']:>7.2f}s  "
                  f"{r['min_s']:>7.2f}s  {r['max_s']:>7.2f}s  {r['tokens_per_sec']:>7.1f}  {r['peak_mem_gb']:>7.1f}GB")
        else:
            print(f"{r['n_images']:>8}  {r['status']:>6}  {'--':>9}  {'--':>8}  {'--':>8}  {'--':>8}  {'--':>8}  {r['peak_mem_gb']:>7.1f}GB")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark generate_response() wall-clock latency vs. number of images in a single multi-image conversation."
    )
    parser.add_argument("--model_name", type=str, default="qwen2.5-vl-3b")
    parser.add_argument("--image_counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64,
                         help="Tokens generated per sample. Generation is deterministic "
                              "(do_sample=False) with min_new_tokens pinned to this value too, "
                              "so every call does exactly the same amount of decode work and "
                              "tokens_per_sec is comparable across N.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--focus", action="store_true",
                         help="Benchmark the FOCUS/noise-injection batch path from eval/mantis.py "
                              "(N+1 noise-variant conversations per call) instead of the plain "
                              "N-images-in-one-conversation path.")
    parser.add_argument("--noise", type=str, default="imagenet+uniform",
                         choices=["black", "white", "gray", "random", "gaussian", "uniform",
                                  "imagenet+gaussian", "imagenet+uniform", "imagenet+diffusion",
                                  "shot", "impulse", "speckle"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--severity", type=int, choices=[1, 2, 3, 4, 5], default=3)
    parser.add_argument("--hyperparameter", type=float, default=0.03,
                         help="Contrastive decoding weight, only used when --focus is set.")
    args = parser.parse_args()

    set_seed(args.seed)

    run_benchmark(
        model_name=args.model_name,
        image_counts=args.image_counts,
        num_trials=args.num_trials,
        warmup=args.warmup,
        image_size=args.image_size,
        max_new_tokens=args.max_new_tokens,
        focus=args.focus,
        noise_mode=args.noise,
        scale=args.scale,
        severity=args.severity,
        hyperparameter=args.hyperparameter,
        output_csv=args.output_csv,
    )

    if args.output_csv:
        print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
