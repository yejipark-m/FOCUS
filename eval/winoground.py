import os
import re
import argparse
from datasets import load_dataset
from transformers import set_seed
from tqdm import tqdm
from model.model_handler import ModelHandler
from utils.cd_sample import evolve_cd_sampling
from utils.add_noise import NoiseProcessor


def evaluate_winoground(model_name, dataset, hyperparameter, noise):
    """
    Evaluates the specified vision-language model on the Winoground dataset.
    """
    # Initialize the model handler
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        hyperparameter = float(hyperparameter)
        evolve_cd_sampling(hyperparameter)  # hooks sample.hyperparameter internally
    model_handler.model.eval()

    # Category-wise scores
    scores = {'text': 0, 'image': 0, 'group': 0, 'total': 0}

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        # Load images and captions
        image_0, image_1 = example["image_0"].convert("RGB"), example["image_1"].convert("RGB")
        caption_0, caption_1 = example["caption_0"], example["caption_1"]

        # Instruction for Caption Matching
        system_prompt = "You are an evaluator tasked with comparing images and captions " \
                        "to answer the question."

        text_score_instructions = f"Does this image depict: (A): {caption_0} or (B): {caption_1}? Answer with A or B."
        image_score_instructions = f"Which image better aligns with the description: <caption_placeholder>? " \
                                     f"The (A): first or the (B): second image? Answer with A or B."

        img_instructions_0 = image_score_instructions.replace("<caption_placeholder>", caption_0)
        img_instructions_1 = image_score_instructions.replace("<caption_placeholder>", caption_1)

        text_i0_image = image_0
        text_i1_image = image_1
        images = [image_0, image_1]

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

        predictions = {
            "text_i0": model_handler.generate_response([text_i0_image], system_prompt=system_prompt, instruction=text_score_instructions).strip(),
            "text_i1": model_handler.generate_response([text_i1_image], system_prompt=system_prompt, instruction=text_score_instructions).strip(),
            "image_t0": model_handler.generate_response(images, system_prompt=system_prompt, instruction=img_instructions_0).strip(),
            "image_t1": model_handler.generate_response(images, system_prompt=system_prompt, instruction=img_instructions_1).strip()
        }

        ground_truths = {
            "text_i0": "a",
            "text_i1": "b",
            "image_t0": "a",
            "image_t1": "b"
        }

        # Get predictions
        for key in predictions:
            if model_name == "qwen3-vl-32b-thinking":
                predictions[key] = predictions[key].split("</think>\n\n")[-1].strip()
            pred = predictions[key].lower().strip()
            pred = pred.replace(",", "").replace(".", "").strip()

            # Match 'a' or 'b' as a standalone word
            match = re.search(r'\b([ab])\b', pred)

            if match and match.group(1) in ["a", "b"]:
                predictions[key] = match.group(1)
            else:
                predictions[key] = "a"  # Default or fallback

        # Update scores
        scores['total'] += 1

        if predictions["text_i0"] == ground_truths["text_i0"] and predictions["text_i1"] == ground_truths["text_i1"]:
            scores['text'] += 1

        if predictions["image_t0"] == ground_truths["image_t0"] and predictions["image_t1"] == ground_truths[
            "image_t1"]:
            scores['image'] += 1

        if (predictions["text_i0"] == ground_truths["text_i0"] and predictions["text_i1"] == ground_truths["text_i1"])\
                and\
                (predictions["image_t0"] == ground_truths["image_t0"] and predictions["image_t1"] == ground_truths[
            "image_t1"]):
            scores['group'] += 1

    # Print category-wise results
    print(f"\n=== Evaluation ===")

    total = scores['total']

    text_acc = scores['text'] / total
    image_acc = scores['image'] / total
    group_acc = scores['group'] / total

    print(f"Text: {text_acc:.4f}")
    print(f"Image: {image_acc:.4f}")
    print(f"Group: {group_acc:.4f}")


if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with Winoground.")
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
    winoground = load_dataset('facebook/winoground', split='test')

    if args.split == 'val':
        winoground = winoground.shuffle(args.seed).select(range(int(0.1 * len(winoground))))

    print("Dataset loaded successfully.")

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

    # Evaluate the model
    evaluate_winoground(model_name=args.model_name, dataset=winoground, hyperparameter=args.hyperparameter, noise=noise_processor)
