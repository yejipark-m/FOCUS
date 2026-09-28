import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoProcessor, AutoTokenizer
from configs.model_registry import MODEL_CONFIGS


class ModelHandler:
    def __init__(self, model_name: str, device: str = "cuda"):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")

        config = MODEL_CONFIGS[model_name]
        self.device = device
        self.model_name = model_name

        model_class = config["class"]
        pretrained_path = config["pretrained_path"]
        load_kwargs = {
            **config.get("load_kwargs", {})
        }
        # Load base model
        self.model = model_class.from_pretrained(pretrained_path, device_map="auto", **load_kwargs)

        # Load processor and tokenizer
        processor_path = config.get("processor_pretrained_path", pretrained_path)
        self.processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_path)
        if "llava-onevision" in model_name:
            self.processor.tokenizer.padding_side = "left"

    def generate_response(
            self,
            image,
            system_prompt: Optional[str] = None,
            instruction: Optional[str] = "Describe this image."):

        prompt = self._build_prompt(image, instruction, system_prompt)
        inputs = self.processor(images=image, text=prompt, padding=True, return_tensors="pt").to(device=self.device, dtype=torch.bfloat16)

        # Perform a single forward pass through the model
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512)
        return self._decode_output(output_ids, inputs)

    def generate_logits(
            self,
            image,
            system_prompt: Optional[str] = None,
            instruction: Optional[str] = "Describe this image.",
            return_prompt: bool = False,
    ):
        """
        Returns logits from the model for the last step of the prompt.
        """
        prompt = self._build_prompt(image, instruction, system_prompt)
        inputs = self.processor(images=image, text=prompt, padding=True, return_tensors="pt").to(device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)

        # Get the logits of the last text token (before any generation)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)
        last_logits = logits[:, -1, :].float()  # shape: (vocab_size,)
        return last_logits

    def generate_cot(
            self,
            image,
            system_prompt: Optional[str] = None,
            instruction: Optional[str] = "Describe this image."
    ):
        save_result = ""
        base_instruction = "Describe this image."
        for i, img in enumerate(image):
            order_prompt = f"This is image {i+1} of {len(image)}."
            prompt = self._build_prompt([img], base_instruction + order_prompt, system_prompt)
            inputs = self.processor(images=img,
                                    text=prompt,
                                    padding=True,
                                    return_tensors="pt"
                                    ).to(device=self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512)
            save_result += self._decode_output(output_ids, inputs)
        cot_instruction = instruction + " Let's think step by step." + save_result
        cot_prompt = self._build_prompt(images=None, instruction=cot_instruction, system_prompt=system_prompt)
        cot_inputs = self.processor(images=None, text=cot_prompt, padding=True, return_tensors="pt").to(device=self.device,
                                                                                                 dtype=torch.bfloat16)

        # Perform a single forward pass through the model
        with torch.no_grad():
            output_ids = self.model.generate(
                **cot_inputs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=8192)

        return self._decode_output(output_ids, cot_inputs)

    def _build_prompt(self, images, instruction, system_prompt=None):
        """
        Builds one or more prompts depending on whether `images` is a single list or a list of lists.

        Args:
            images (List[Image] or List[List[Image]] or None): One or more sets of images.
            instruction (str): User instruction text.
            system_prompt (str, optional): Optional system message.

        Returns:
            List[str]: List of chat prompts formatted via apply_chat_template.
        """

        def build_single_conversation(img_list):
            conversation = []
            user_content = []

            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })

            if img_list is not None:
                user_content = [{"type": "image", "image": img} for img in img_list]
            user_content.append({"type": "text", "text": instruction})

            conversation.append({
                "role": "user",
                "content": user_content
            })

            return conversation

        # Determine if images is a batch (list of lists) or a single list
        is_batched = isinstance(images, list) and all(isinstance(sub, list) for sub in images)

        if is_batched:
            return [
                self.processor.apply_chat_template(
                    build_single_conversation(image_group),
                    add_generation_prompt=True
                )
                for image_group in images
            ]
        else:
            return self.processor.apply_chat_template(
                    build_single_conversation(images),
                    add_generation_prompt=True)

    def _get_image_token_id(self):
        """Return the token ID used as image placeholder in the input sequence."""
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index", "img_token_id"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise NotImplementedError(f"Cannot find image_token_id in config for {self.model_name}")

    def _image_token_counts(self, grid_thw):
        """Tokens per image in LLM sequence (after spatial merger)."""
        try:
            merge = self.model.visual.merger.spatial_merge_size
        except AttributeError:
            merge = 2
        counts = [int(t * (h // merge) * (w // merge)) for t, h, w in grid_thw.tolist()]
        return counts

    def _capture_lm_head_input(self, inputs):
        """Run forward pass and capture the input to lm_head via a hook."""
        captured = {}
        hook = self.model.lm_head.register_forward_pre_hook(
            lambda m, inp: captured.update({"hidden": inp[0].detach()})
        )
        with torch.no_grad():
            self.model(**inputs, return_dict=True)
        hook.remove()
        return captured["hidden"][0].float()

    def get_lm_visual_tokens(self, image):
        """
        Run full forward pass (single image) and return lm_head-input features
        at image token positions. Shape: (num_image_tokens, lm_hidden), L2-normalized.
        """
        if hasattr(self.model, "visual"):
            conversation = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": "Describe the image."}]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
            inputs = self.processor(images=[image], text=text, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=image, text="Describe the image.", return_tensors="pt").to(self.device)

        hidden = self._capture_lm_head_input(inputs)         # (seq_len, lm_hidden)
        img_id = self._get_image_token_id()
        mask = (inputs["input_ids"][0] == img_id).to(hidden.device)
        return F.normalize(hidden[mask], dim=-1)

    def get_lm_visual_tokens_multi(self, images: list):
        """
        Run full forward pass (multi-image) and return lm_head-input features
        split per image. Returns list of tensors (num_tokens_i, lm_hidden), L2-normalized.
        """
        if hasattr(self.model, "visual"):
            conversation = [{"role": "user", "content":
                [{"type": "image", "image": img} for img in images] +
                [{"type": "text", "text": "Describe the image."}]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
            inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
            token_counts = self._image_token_counts(inputs["image_grid_thw"])
            img_id = self._get_image_token_id()
            total_img_toks = int((inputs["input_ids"][0] == img_id).sum())
            if sum(token_counts) != total_img_toks:
                n_each = total_img_toks // len(images)
                token_counts = [n_each] * len(images)
        else:
            inputs = self.processor(images=images, text="Describe the image.", return_tensors="pt").to(self.device)
            img_id = self._get_image_token_id()
            total_img_toks = int((inputs["input_ids"][0] == img_id).sum())
            token_counts = [total_img_toks // len(images)] * len(images)

        hidden = self._capture_lm_head_input(inputs)         # (seq_len, lm_hidden)
        img_id = self._get_image_token_id()
        mask = (inputs["input_ids"][0] == img_id).to(hidden.device)
        img_hidden = hidden[mask]                              # (total_img_toks, lm_hidden)

        result, offset = [], 0
        for n in token_counts:
            chunk = img_hidden[offset: offset + n]
            result.append(F.normalize(chunk, dim=-1))
            offset += n
        return result

    def _decode_output(self, output_ids, inputs):
        return self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
