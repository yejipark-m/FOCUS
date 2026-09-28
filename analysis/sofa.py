"""
SoFt Attention (SoFA) – training-free position-bias mitigation for multi-image LVLMs.

Reference:
    Tian et al., "Identifying and Mitigating Position Bias of Multi-image
    Vision-Language Models", CVPR 2025.

Core idea (Eq. 2 in the paper):
    M_soft = (1 - σ) * M_causal + σ * M_bidirectional

Only inter-image attention is modified; text-to-text attention remains causal.
SoFA is applied every `apply_every_n_layers` layers (default 2) as in the paper.

Usage
-----
from model.model_handler import ModelHandler
from analysis.sofa import SoFAWrapper

handler = ModelHandler("llava-onevision-7b")
wrapper = SoFAWrapper(handler, sigma=0.5)

response = wrapper.generate(images, instruction="...")
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def _image_token_ranges(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> List[Tuple[int, int]]:
    """Return (start, end) [end exclusive] for each contiguous run of image tokens."""
    ids = input_ids[0].tolist()
    ranges, i = [], 0
    while i < len(ids):
        if ids[i] == image_token_id:
            start = i
            while i < len(ids) and ids[i] == image_token_id:
                i += 1
            ranges.append((start, i))
        else:
            i += 1
    return ranges


def build_sofa_mask(
    seq_len: int,
    image_ranges: List[Tuple[int, int]],
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    big_neg: float = -1e4,
) -> torch.Tensor:

    # Standard causal additive mask
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    mask[upper] = big_neg

    if len(image_ranges) < 2 or sigma == 0.0:
        return mask.unsqueeze(0).unsqueeze(0)

    # Boolean mask: True where token is an image token
    is_img = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for s, e in image_ranges:
        is_img[s:e] = True

    # Positions normally blocked by causal mask
    inter_blocked = upper & is_img.unsqueeze(0) & is_img.unsqueeze(1)

    # Replace big_neg with the interpolated value
    # M_soft[i,j] = (1-σ)*M_causal[i,j] + σ*M_bidir[i,j]
    # Additive equivalent: (1-σ)*big_neg + σ*0 = (1-σ)*big_neg
    mask[inter_blocked] = (1.0 - sigma) * big_neg

    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, L, L)


# ---------------------------------------------------------------------------
# Layer finder
# ---------------------------------------------------------------------------

def _get_llm_attn_layers(model) -> List[torch.nn.Module]:
    """Return a list of self-attention modules from the LLM backbone."""
    search_paths = [
        "language_model.model.layers",   # LlavaOnevision, InternVL
        "model.language_model.layers",   # Qwen2.5-VL
        "model.layers",                  # generic fallback
        "model.model.layers",            # generic fallback
    ]
    for path in search_paths:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            attn_modules = []
            for layer in obj:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn_modules.append(attn)
            if attn_modules:
                return attn_modules
    raise RuntimeError(
        "Cannot locate transformer attention layers. "
        "Add the model's layer path to `_get_llm_attn_layers`."
    )


# ---------------------------------------------------------------------------
# SoFAWrapper
# ---------------------------------------------------------------------------

class SoFAWrapper:
    """
    Wraps a ModelHandler and injects SoFA into generation.

    Parameters
    ----------
    handler : ModelHandler
        An already-loaded ModelHandler instance.
    sigma : float
        Interpolation weight toward bidirectional attention (0 = causal, 1 = bidir).
        Paper default / typical range: 0.25 – 0.75 (tune on a 32-shot val set).
    apply_every_n_layers : int
        Apply SoFA on every Nth layer; others keep standard causal attention.
        Paper uses 2 (i.e., layers 0, 2, 4, …).
    big_neg : float
        The "negative infinity" used in the additive mask.  Should be a large
        negative float rather than -inf to allow interpolation.
    """

    def __init__(
        self,
        handler,
        sigma: float = 0.5,
        apply_every_n_layers: int = 2,
        big_neg: float = -1e4,
    ):
        self.handler = handler
        self.sigma = sigma
        self.apply_every_n_layers = apply_every_n_layers
        self.big_neg = big_neg
        self._hooks: List = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        images,
        instruction: str = "Describe this image.",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        **gen_kwargs,
    ) -> str:
        """Generate a response with SoFA applied."""
        prompt = self.handler._build_prompt(images, instruction, system_prompt)
        inputs = self.handler.processor(
            images=images,
            text=prompt,
            padding=True,
            return_tensors="pt",
        ).to(device=self.handler.device, dtype=torch.bfloat16)

        with self._sofa_context(inputs["input_ids"]):
            with torch.no_grad():
                output_ids = self.handler.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
        return self.handler._decode_output(output_ids, inputs)

    # ------------------------------------------------------------------
    # Context manager that installs / removes hooks
    # ------------------------------------------------------------------

    @contextmanager
    def _sofa_context(self, input_ids: torch.Tensor):
        """Context manager: install SoFA hooks, yield, then remove them."""
        self._install_hooks(input_ids)
        try:
            yield
        finally:
            self._remove_hooks()

    def _install_hooks(self, input_ids: torch.Tensor):
        """Register forward pre-hooks on LLM attention layers."""
        self._remove_hooks()

        try:
            image_token_id = self.handler._get_image_token_id()
        except NotImplementedError:
            # Fallback: no image token found → no SoFA (pure causal)
            return

        seq_len = input_ids.shape[1]
        image_ranges = _image_token_ranges(input_ids, image_token_id)

        if len(image_ranges) < 2:
            return  # nothing to do for single-image inputs

        device = input_ids.device
        attn_layers = _get_llm_attn_layers(self.handler.model)

        for layer_idx, attn_module in enumerate(attn_layers):
            apply_sofa = (layer_idx % self.apply_every_n_layers == 0)
            if not apply_sofa:
                continue

            # Lazily build the mask at hook time so device/dtype are correct
            def _make_hook(sigma, seq_len_, image_ranges_, big_neg_):
                def hook(module, args, kwargs):
                    # Locate the attention_mask in args or kwargs
                    # Most transformers attention modules take it as the 2nd positional arg
                    # or as keyword arg 'attention_mask'.
                    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                        existing = kwargs["attention_mask"]
                        dev, dt = existing.device, existing.dtype
                        sofa = build_sofa_mask(
                            seq_len_, image_ranges_, sigma, dev, dt, big_neg_
                        )
                        # Broadcast to match batch size if needed
                        if existing.shape[0] > 1:
                            sofa = sofa.expand(existing.shape[0], -1, -1, -1)
                        kwargs["attention_mask"] = sofa
                    elif len(args) >= 2 and args[1] is not None:
                        existing = args[1]
                        dev, dt = existing.device, existing.dtype
                        sofa = build_sofa_mask(
                            seq_len_, image_ranges_, sigma, dev, dt, big_neg_
                        )
                        if existing.shape[0] > 1:
                            sofa = sofa.expand(existing.shape[0], -1, -1, -1)
                        args = (args[0], sofa) + args[2:]
                    return args, kwargs

                return hook

            h = attn_module.register_forward_pre_hook(
                _make_hook(self.sigma, seq_len, image_ranges, self.big_neg),
                with_kwargs=True,
            )
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()