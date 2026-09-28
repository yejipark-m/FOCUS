import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import set_seed
from tqdm import tqdm
from model.model_handler import ModelHandler
from utils.cd_sample import evolve_cd_sampling
from utils.add_noise import NoiseProcessor
from analysis.sofa import SoFAWrapper


def get_task_instruction(dataset):
    if dataset in ['analogy', 'attribute', 'plot_code', 'visual_chain', 'plot_text', 'sightseeing',
                   'image_needles_concat']:
        instr = 'Answer with a single word.'
    elif dataset in ['codeu', 'food', 'image_jigsaw', 'codeu_text']:
        instr = 'Answer with the option symbol.'
    elif dataset in ['arxiv', 'arxiv_text']:
        instr = 'Answer with the paper title.'
    elif dataset in ['count', 'count_concat']:
        instr = 'Answer with a single number.'
    elif dataset in ['3d_scene', '3d_scene_concat']:
        instr = 'The following images are different views of the same 3D scene. Answer with a single number.'

    return instr


def evaluate_mirb(model_name, dataset, hyperparameter, noise, mode="cot", sigma=0.5):
    """
    Evaluates the specified vision-language model on the MIRB dataset.

    mode : "cot"      – generate_cot (caption each image separately, then answer)
           "response" – generate_response (pass all images together)
           "sofa"     – generate_response with SoFt Attention (Tian et al., CVPR 2025)
    sigma : interpolation weight for SoFA (only used when mode="sofa")
    """
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        hyperparameter = float(hyperparameter)
        evolve_cd_sampling(hyperparameter)
    model_handler.model.eval()

    if mode == "sofa":
        generator = SoFAWrapper(model_handler, sigma=sigma)
    else:
        generator = model_handler

    # Global and subgroup tracking
    scores = {'acc': 0, 'total': 0}
    subtask_scores = defaultdict(lambda: {'acc': 0, 'total': 0})
    image_count_scores = defaultdict(lambda: {'acc': 0, 'total': 0})

    for example in tqdm(dataset, desc=f"Evaluating {model_name} [{mode}]"):
        images = [img.convert("RGB").resize((512, 512)) for img in example["image_list"]]
        num_images = len(images)

        if noise is not None:
            pure_noise, single_noise = noise
            pure_noise_images = [pure_noise(img) for img in images]
            injected_noise_images = [
                [img if j == i else single_noise(images[j]) for j, img in enumerate(images)]
                for i in range(len(images))
            ]
            images = [pure_noise_images, *injected_noise_images]

        question = example['questions']
        subset_task_name = example['subset']
        instruction = get_task_instruction(subset_task_name)
        prompt = instruction + question

        if mode == "cot":
            predictions = generator.generate_cot(images, instruction=prompt).strip()
        elif mode == "response":
            predictions = generator.generate_response(images, instruction=prompt).strip()
        elif mode == "sofa":
            predictions = generator.generate(images, instruction=prompt).strip()

        ground_truths = example["answers"]

        if model_name == "qwen3-vl-32b-thinking":
            predictions = predictions.split("</think>\n\n")[-1].strip()
        prediction = predictions.lower().strip().replace(",", "").replace(".", "")
        ground_truth = ground_truths.lower().strip().replace("(", "").replace(")", "")

        # Update scores
        scores['total'] += 1
        subtask_scores[subset_task_name]['total'] += 1
        image_count_scores[num_images]['total'] += 1

        if prediction == ground_truth:
            scores['acc'] += 1
            subtask_scores[subset_task_name]['acc'] += 1
            image_count_scores[num_images]['acc'] += 1

    # Compute final results
    total = scores['total']
    acc = scores['acc'] / total
    per_subtask_acc = {k: v['acc'] / v['total'] for k, v in subtask_scores.items()}
    per_image_count_acc = {k: v['acc'] / v['total'] for k, v in sorted(image_count_scores.items())}

    return acc, per_subtask_acc, per_image_count_acc



if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with MIRB.")
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
        "--mode",
        choices=["cot", "response", "sofa"], default="response",
        help=(
            "cot      – caption each image separately, then answer;\n"
            "response – pass all images together in one forward pass;\n"
            "sofa     – response mode with SoFt Attention (Tian et al., CVPR 2025)."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float, default=0.5,
        help="SoFA interpolation weight σ ∈ [0,1] (only used with --mode sofa). "
             "0 = causal, 1 = bidirectional. Paper tunes on a 32-shot val set.",
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
    dataset_name = 'VLLMs/MIRB-hf'

    all_results = {}
    data = load_dataset(dataset_name, split="test")

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

    acc, per_subtask_acc, per_image_count_acc = evaluate_mirb(
        model_name=args.model_name,
        dataset=data,
        hyperparameter=args.hyperparameter,
        noise=noise_processor,
        mode=args.mode,
        sigma=args.sigma,
    )

    print(f"\n=== Overall Accuracy  [{args.mode}] ===")
    print(f"ACC: {acc:.4f}")

    print("\n=== Accuracy per Subtask ===")
    for task, a in per_subtask_acc.items():
       print(f"{task}: {a:.4f}")

    print("\n=== Accuracy per Image Count ===")
    for count, a in per_image_count_acc.items():
        print(f"{count} images: {a:.4f}")
