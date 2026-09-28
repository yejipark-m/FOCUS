import re
import time
import argparse
from datasets import load_dataset
from transformers import set_seed
from tqdm import tqdm
from model.model_handler import ModelHandler
from utils.cd_sample import evolve_cd_sampling
from utils.add_noise import NoiseProcessor


def parse_answer(raw_answer):
    if "final answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("final answer:") + len("final answer:"):].strip()
    elif "the answer is" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("the answer is") + len("the answer is"):].strip()
    elif "answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("answer:") + len("answer:"):].strip()
    elif "prediction:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("prediction:") + len("prediction:"):].strip()
    else:
        answer = raw_answer
    return answer


def evaluate_mantis(model_name, dataset, hyperparameter, noise):
    """
    Evaluates the specified vision-language model on the Mantis dataset.
    """
    # Initialize the model handler
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        hyperparameter = float(hyperparameter)
        evolve_cd_sampling(hyperparameter)  # hooks sample.hyperparameter internally
    model_handler.model.eval()

    # Category-wise scores
    scores = {'acc': 0, 'total': 0}
    sample_records = []  # (num_images, latency_sec, correct)

    wall_start = time.perf_counter()

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        sample_start = time.perf_counter()
        images = [img.convert("RGB") for img in example["images"]]
        num_images_orig = len(images)

        if noise is not None:
            pure_noise = noise[0]
            single_noise = noise[1]

            # Pure noise applied to all images
            pure_noise_images = [pure_noise(img) for img in images]

            # Inject noise into all positions (one noisy image at a time)
            injected_noise_images = [
                [img if j == i else single_noise(images[j]) for j, img in enumerate(images)]
                for i in range(len(images))
            ]

            # Combine original, all-pure-noise, and each injected variant
            images = [pure_noise_images, *injected_noise_images]

        # Instruction for Caption Matching
        question_type, question, option = example["question_type"], example["question"], example["options"]

        if "<image>" in question:
            question = question.replace("<image>", "")

        # Question
        if question_type == 'short-answer':
            prompt = f'Given the images, answer the following short answer vqa question:\nQ: {question}\nYou can first give your analysis, and then give your final answer as "Final Answer:"'
        if question_type == 'multi-choice':
            prompt = f"{question}\nAnswer with the option's letter from the given choices directly. {option}"

        t0 = time.perf_counter()
        predictions = model_handler.generate_response(images, instruction=prompt).strip()
        latency = time.perf_counter() - t0

        ground_truths = example["answer"]

        prediction = parse_answer(predictions.lower().strip())
        prediction = prediction.replace(",", "").replace(".", "")

        match = re.search(r'\(?([abcd])\)?', prediction)
        pred_letter = match.group(1) if match else "abcd"

        ground_truth = ground_truths.lower().strip().replace("(", "").replace(")", "")

        correct = pred_letter == ground_truth
        scores['total'] += 1
        if correct:
            scores['acc'] += 1

        sample_wall = time.perf_counter() - sample_start
        sample_records.append((num_images_orig, latency, sample_wall, correct))

        cumulative_acc = scores['acc'] / scores['total']
        elapsed = time.perf_counter() - wall_start
        tqdm.write(
            f"  [#{scores['total']:4d}] images={num_images_orig}  latency={latency:.2f}s  "
            f"correct={correct}  running_acc={cumulative_acc:.4f}  elapsed={elapsed:.1f}s"
        )

    wall_elapsed = time.perf_counter() - wall_start
    total = scores['total']
    acc = scores['acc'] / total

    latencies = [r[1] for r in sample_records]
    latencies_sorted = sorted(latencies)
    mean_lat = sum(latencies) / len(latencies)
    p50 = latencies_sorted[int(0.50 * len(latencies_sorted))]
    p90 = latencies_sorted[int(0.90 * len(latencies_sorted))]

    print(f"\n=== Evaluation Summary ===")
    print(f"Total samples : {total}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Wall time     : {wall_elapsed:.1f}s  ({wall_elapsed/total:.2f}s/sample)")
    print(f"Latency (gen) : mean={mean_lat:.2f}s  p50={p50:.2f}s  p90={p90:.2f}s")

    from collections import defaultdict
    by_n = defaultdict(list)
    for n_img, lat, wall, correct in sample_records:
        by_n[n_img].append((lat, wall, correct))

    print(f"\n{'#img':>5}  {'n':>5}  {'acc':>6}  {'lat_mean':>9}  {'lat_std':>8}  {'lat_p50':>8}  {'wall_mean':>10}  {'wall_std':>9}  {'wall_p50':>9}")
    for n_img in sorted(by_n):
        rows = by_n[n_img]
        lats = sorted(r[0] for r in rows)
        walls = sorted(r[1] for r in rows)
        group_acc = sum(r[2] for r in rows) / len(rows)
        lat_mean = sum(lats) / len(lats)
        wall_mean = sum(walls) / len(walls)
        lat_std = (sum((x - lat_mean) ** 2 for x in lats) / len(lats)) ** 0.5
        wall_std = (sum((x - wall_mean) ** 2 for x in walls) / len(walls)) ** 0.5
        print(
            f"{n_img:>5}  {len(rows):>5}  {group_acc:>6.4f}"
            f"  {lat_mean:>8.2f}s"
            f"  {lat_std:>7.2f}s"
            f"  {lats[int(0.50*len(lats))]:>7.2f}s"
            f"  {wall_mean:>9.2f}s"
            f"  {wall_std:>8.2f}s"
            f"  {walls[int(0.50*len(walls))]:>8.2f}s"
        )

    return acc


if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with Mantis.")
    parser.add_argument(
        "--model_name",
        type=str, default="qwen2.5-vl-3b",
        help="Supported models: llava-onevision, qwen2.5-vl, internvl3",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"], default="test",
    )
    parser.add_argument(
        "--hyperparameter", default=None,
    )
    parser.add_argument(
        "--noise",
        choices=[
            "black", "white", "gray", "random", "gaussian", "uniform",
            "imagenet+gaussian", "imagenet+uniform", "imagenet+diffusion", "shot", "impulse", "speckle",
        ],
        default="imagenet+uniform"
    )
    parser.add_argument(
        "--scale",
        type=float, default=1,
        help="Scale for the image augmentation process."
    )
    parser.add_argument(
        "--severity",
        type=int, choices=[1, 2, 3, 4, 5], default=3,
        help="Severity level for noise types that require it (e.g., shot, impulse, speckle)."
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load dataset
    dataset_name = 'TIGER-Lab/Mantis-Eval'

    all_results = {}
    data = load_dataset(dataset_name, split='test')

    if args.split == 'val':
        data = data.shuffle(args.seed).select(range(int(0.1 * len(data))))

    if args.hyperparameter is not None:
        pure_noise_processor = NoiseProcessor(
            mode=args.noise, scale=1.0, severity=args.severity
        )
        single_noise_processor = NoiseProcessor(
            mode=args.noise, scale=args.scale, severity=args.severity
        )
        noise_processor = [pure_noise_processor, single_noise_processor]
    else:
        noise_processor = None

    acc = evaluate_mantis(model_name=args.model_name, dataset=data, hyperparameter=args.hyperparameter, noise=noise_processor)
    print(f"ACC: {acc:.4f}")