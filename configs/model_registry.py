import torch
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    InternVLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

# Configuration for supported models

MODEL_CONFIGS = {
    "llava-onevision-0.5b": {
        "class": LlavaOnevisionForConditionalGeneration,
        "pretrained_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    },
    "llava-onevision-7b":{
        "class": LlavaOnevisionForConditionalGeneration,
        "pretrained_path": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
         "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
         }
    },
    "qwen2.5-vl-3b": {
        "class": Qwen2_5_VLForConditionalGeneration,
        "pretrained_path": "Qwen/Qwen2.5-VL-3B-Instruct",
         "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
         }
    },
    "qwen2.5-vl-7b": {
        "class": Qwen2_5_VLForConditionalGeneration,
        "pretrained_path": "Qwen/Qwen2.5-VL-7B-Instruct",
         "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
         }
    },
    "qwen3-vl-4b": {
        "class": Qwen3VLForConditionalGeneration,
        "pretrained_path": "Qwen/Qwen3-VL-4B-Instruct",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    },
    "qwen3-vl-8b": {
        "class": Qwen3VLForConditionalGeneration,
        "pretrained_path": "Qwen/Qwen3-VL-8B-Instruct",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    },
    "qwen3-vl-30b-a3b": {
        "class": Qwen3VLMoeForConditionalGeneration,
        "pretrained_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    },
    "qwen3-vl-32b": {
	"class": Qwen3VLForConditionalGeneration,
	"pretrained_path": "Qwen/Qwen3-VL-32B-Instruct",
	"load_kwargs": {
	    "torch_dtype": torch.bfloat16,
	    "attn_implementation": "flash_attention_2"
        }
    },
    "internvl3-2b": {
        "class": InternVLForConditionalGeneration,
        "pretrained_path": "OpenGVLab/InternVL3-2B-hf",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    },
    "internvl3-8b": {
        "class": InternVLForConditionalGeneration,
        "pretrained_path": "OpenGVLab/InternVL3-8B-hf",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    },
    "qwen3-vl-32b-thinking": {
        "class": Qwen3VLMoeForConditionalGeneration,
        "pretrained_path": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "load_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        }
    }
}
