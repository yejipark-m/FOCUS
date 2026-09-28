import os
import re
import argparse
from datasets import load_dataset
from transformers import set_seed
from tqdm import tqdm
from model.model_handler import ModelHandler
from utils.cd_sample import evolve_cd_sampling
from utils.add_noise import NoiseProcessor
from utils.video_utils import sample_frames


def evaluate_vinoground(model_name, dataset, video_dir, hyperparameter, noise, num_frames=8):
    """
    Evaluates the specified vision-language model on the Vinoground dataset.

    Vinoground is the video analogue of Winoground: each example contains two short
    video clips (pos / neg) and two captions (pos_cap / neg_cap), where the
    correct pairing is pos ↔ pos_cap and neg ↔ neg_cap.

    Metrics:
      Text  – both videos individually identified their correct caption.
      Video – given each caption, the correct video was selected from a side-by-side pair.
      Group – both text and video scores are correct simultaneously.
    """
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        evolve_cd_sampling(float(hyperparameter))
    model_handler.model.eval()

    system_prompt = (
        "You are an evaluator tasked with comparing video clips and captions "
        "to answer the question."
    )

    scores = {"text": 0, "video": 0, "group": 0, "total": 0}

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        caption_0, caption_1 = example["pos_cap"], example["neg_cap"]
        idx = example["index"]
        frames_0 = sample_frames(os.path.join(video_dir, f"{idx}_pos.mp4"), num_frames)
        frames_1 = sample_frames(os.path.join(video_dir, f"{idx}_neg.mp4"), num_frames)
        n = len(frames_0)

        # Text score: given one video's frames, which caption matches?
        text_instruction = (
            f"Does this video show: (A): {caption_0} or (B): {caption_1}? Answer with A or B."
        )

        # Video score: given a caption, which of the two video clips matches?
        # First n frames = Video A, next n frames = Video B.
        video_instruction_t0 = (
            f"The first {n} frames are from Video A and the next {n} frames are from Video B. "
            f'Which video better depicts: "{caption_0}"? Answer with A or B.'
        )
        video_instruction_t1 = (
            f"The first {n} frames are from Video A and the next {n} frames are from Video B. "
            f'Which video better depicts: "{caption_1}"? Answer with A or B.'
        )

        combined_frames = frames_0 + frames_1

        if noise is not None:
            pure_noise, single_noise = noise

            # Per-video text inputs with noise injection (mirrors winoground pattern)
            def inject_noise(frames):
                pure = [pure_noise(f) for f in frames]
                injected = [
                    [f if j == i else single_noise(frames[j]) for j, f in enumerate(frames)]
                    for i in range(len(frames))
                ]
                return [pure, *injected]

            text_i0_input = inject_noise(frames_0)
            text_i1_input = inject_noise(frames_1)
            video_input = inject_noise(combined_frames)
        else:
            text_i0_input = frames_0
            text_i1_input = frames_1
            video_input = combined_frames

        raw = {
            "text_v0": model_handler.generate_response(
                text_i0_input, system_prompt=system_prompt, instruction=text_instruction
            ).strip(),
            "text_v1": model_handler.generate_response(
                text_i1_input, system_prompt=system_prompt, instruction=text_instruction
            ).strip(),
            "video_t0": model_handler.generate_response(
                video_input, system_prompt=system_prompt, instruction=video_instruction_t0
            ).strip(),
            "video_t1": model_handler.generate_response(
                video_input, system_prompt=system_prompt, instruction=video_instruction_t1
            ).strip(),
        }

        ground_truths = {"text_v0": "a", "text_v1": "b", "video_t0": "a", "video_t1": "b"}
        predictions = {}

        for key, resp in raw.items():
            if model_name == "qwen3-vl-32b-thinking":
                resp = resp.split("</think>\n\n")[-1].strip()
            pred = resp.lower().strip().replace(",", "").replace(".", "")
            match = re.search(r'\b([ab])\b', pred)
            predictions[key] = match.group(1) if match else "a"

        scores["total"] += 1
        text_correct = (
            predictions["text_v0"] == ground_truths["text_v0"]
            and predictions["text_v1"] == ground_truths["text_v1"]
        )
        video_correct = (
            predictions["video_t0"] == ground_truths["video_t0"]
            and predictions["video_t1"] == ground_truths["video_t1"]
        )
        if text_correct:
            scores["text"] += 1
        if video_correct:
            scores["video"] += 1
        if text_correct and video_correct:
            scores["group"] += 1

    total = scores["total"]
    print(f"\n=== Vinoground Evaluation ===")
    print(f"Text  : {scores['text'] / total:.4f}")
    print(f"Video : {scores['video'] / total:.4f}")
    print(f"Group : {scores['group'] / total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with Vinoground.")
    parser.add_argument(
        "--model_name",
        type=str, default="qwen2.5-vl-3b",
        help="Supported models: llava-onevision, qwen2.5-vl, internvl3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--video_dir",
        type=str, default="/root/share/Vinoground/vinoground_videos",
        help="Directory containing {index}_pos.mp4 and {index}_neg.mp4 files.",
    )
    parser.add_argument(
        "--num_frames",
        type=int, default=8,
        help="Number of frames to sample uniformly from each video clip.",
    )
    parser.add_argument("--hyperparameter", default=None)
    parser.add_argument(
        "--noise",
        choices=[
            "black", "white", "gray", "random", "gaussian", "uniform",
            "imagenet+gaussian", "imagenet+uniform", "imagenet+diffusion",
            "shot", "impulse", "speckle",
        ],
        default="imagenet+uniform",
    )
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--severity", type=int, choices=[1, 2, 3, 4, 5], default=3)

    args = parser.parse_args()
    set_seed(args.seed)

    dataset = load_dataset("HanSolo9682/Vinoground", split="test")
    if args.split == "val":
        dataset = dataset.shuffle(args.seed).select(range(int(0.1 * len(dataset))))

    print("Dataset loaded successfully.")

    if args.hyperparameter is not None:
        noise_processor = [
            NoiseProcessor(mode=args.noise, scale=1.0, severity=args.severity),
            NoiseProcessor(mode=args.noise, scale=args.scale, severity=args.severity),
        ]
    else:
        noise_processor = None

    evaluate_vinoground(
        model_name=args.model_name,
        dataset=dataset,
        video_dir=args.video_dir,
        hyperparameter=args.hyperparameter,
        noise=noise_processor,
        num_frames=args.num_frames,
    )
