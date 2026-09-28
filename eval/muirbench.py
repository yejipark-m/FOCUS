import re
import argparse
import torch
from collections import defaultdict
from datasets import load_dataset
from transformers import set_seed
from tqdm import tqdm
from model.model_handler import ModelHandler
from utils.cd_sample import evolve_cd_sampling
from utils.add_noise import NoiseProcessor


def evaluate_muirbench(model_name, dataset, hyperparameter, noise):
    """
    Evaluates the specified vision-language model on the MuirBench dataset.
    """

    # Initialize the model handler
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        hyperparameter = float(hyperparameter)
        evolve_cd_sampling(hyperparameter)  # hooks sample.hyperparameter internally
    model_handler.model.eval()

    # Category-wise scores
    scores = {'acc': 0, 'total': 0}
    image_count_scores = defaultdict(lambda: {'acc': 0, 'total': 0})

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        images = [img.convert("RGB").resize((512,512)) for img in example["image_list"]]
        num_images = len(images)

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
        question = example['question']
        if "<image>" in question:
            question = question.replace("<image>", "")
        choices = example['options']
        if "<image>" in choices:
            choices = question.replace("<image>", "")

        # Question
        question_text = f"Question: {question}"

        # Choices
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A') + i)}) {choice}")
        choices_text = "\n".join(texts)

        hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."

        prompt = question_text + choices_text + hint_text

        predictions = model_handler.generate_response(images, instruction=prompt).strip()
        if model_name == "qwen3-vl-32b-thinking":
            predictions = predictions.split("</think>\n\n")[-1].strip()

        ground_truths = example["answer"]

        prediction = predictions.lower().strip()
        prediction = prediction.replace(",", "").replace(".", "")

        match = re.search(r'\(?([abcd])\)?', prediction)
        pred_letter = match.group(1) if match else "abcd"  # default fallback

        ground_truth = ground_truths.lower().strip().replace("(", "").replace(")", "")

        # Update scores
        scores['total'] += 1
        image_count_scores[num_images]['total'] += 1

        if pred_letter == ground_truth:
            scores['acc'] += 1
            image_count_scores[num_images]['acc'] += 1

    total = scores['total']
    acc = scores['acc'] / total
    per_image_count_acc = {k: v['acc'] / v['total'] for k, v in sorted(image_count_scores.items())}  # 👈 NEW

    return acc, per_image_count_acc


if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with MuirBench.")
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
    dataset_name = 'MUIRBENCH/MUIRBENCH'

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

    acc, per_image_count_acc = evaluate_muirbench(
        model_name=args.model_name,
        dataset=data,
        hyperparameter=args.hyperparameter,
        noise=noise_processor)


    # Print results
    print(f"ACC: {acc:.4f}")
    for count, a in per_image_count_acc.items():
        print(f"{count} images: {a:.4f}")