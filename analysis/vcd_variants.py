import argparse

from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed
from model.model_handler import ModelHandler
from utils.add_noise import NoiseProcessor

import os
from typing import Dict, List, Optional, Union
import torch
from torch import nn

import transformers


from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (GenerateNonBeamOutput)


def vcd_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = False
    hyperparameter = getattr(vcd_sample, "hyperparameter", None)

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )


    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue


        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone().float()

        total_batch_size = next_token_logits.shape[0]   # Only apply the method if there are corrupted images

        if total_batch_size != 1:
            noise = next_token_logits[0] # Logits generated from fully noised images
            img = next_token_logits[1:] # Logits from original inputs
            next_token_logits = (img + (hyperparameter * - (img - noise))).expand_as(next_token_logits)
        else:
            pass

        next_token_logits = next_token_logits.to(input_ids.device)


        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

def evolve_vcd_sampling(hyperparameter: Optional[float] = None):
    transformers.generation.utils.GenerationMixin._sample = vcd_sample
    if hyperparameter is not None:
        vcd_sample.hyperparameter = torch.tensor(hyperparameter) if hyperparameter is not None else None

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

def compare_vcd_with_focus(model_name: str, dataset, hyperparameter: float, noise: List | None) -> None:
    """Evaluate *model_name* on the VisMin benchmark with dynamic multi‑image support.
    """
    # Initialise model & decoding hooks
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        hyperparameter = float(hyperparameter)
        evolve_vcd_sampling(hyperparameter)  # hooks sample.hyperparameter internally
    model_handler.model.eval()

    # Global and subgroup tracking
    scores = {'acc': 0, 'total': 0}
    subtask_scores = defaultdict(lambda: {'acc': 0, 'total': 0})
    image_count_scores = defaultdict(lambda: {'acc': 0, 'total': 0})

    for example in tqdm(dataset, desc=f"Evaluating {model_name}"):
        images = [img.convert("RGB").resize((512, 512)) for img in example["image_list"]]
        num_images = len(images)

        if noise is not None:
            # Pure noise applied to all images
            noise_images = [noise(img) for img in images]

            # Combine original, all-pure-noise, and each injected variant
            images = [noise_images, images]

        question = example['questions']
        subset_task_name = example['subset']
        instruction = get_task_instruction(subset_task_name)
        prompt = instruction + question

        predictions = model_handler.generate_response(images, instruction=prompt).strip()
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
    per_image_count_acc = {k: v['acc'] / v['total'] for k, v in sorted(image_count_scores.items())}  # 👈 NEW

    print(f"Overall Accuracy: {acc:.4f}")


if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with VisMin.")
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
    parser.add_argument("--dataset", default="mirb")
    parser.add_argument(
        "--split",
        choices=["val", "test"], default="val",
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
        default="imagenet+diffusion"
    )
    parser.add_argument(
        "--scale",
        type=float, default=0.9,
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
    if args.dataset == "winoground":
        dataset = load_dataset("facebook/winoground", split="test")
        dataset = dataset.shuffle(args.seed).select(range(int(0.1 * len(dataset))))
    elif args.dataset == "vismin":
        dataset = load_dataset('mair-lab/vismin-bench')["test"]
    elif args.dataset == "mirb":
        dataset = load_dataset('VLLMs/MIRB-hf', split="test")

    print("Dataset loaded successfully.")

    if args.hyperparameter is not None:
        single_noise_processor = NoiseProcessor(
            mode=args.noise, scale=args.scale, severity=args.severity
        )
        noise_processor = single_noise_processor
    else:
        noise_processor = None

    # Evaluate the model
    compare_vcd_with_focus(model_name=args.model_name, dataset=dataset, hyperparameter=args.hyperparameter,
                           noise=noise_processor)
