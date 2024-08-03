from typing import Optional, Text

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available

from src.inference_time_alignment.scorers import ImplicitValueScorer
from scripts.utils import get_local_model_path


def get_scorer(
    load_in_4bit: Optional[bool] = False, 
    use_flash_attention_2: Optional[bool] = False,
) -> ImplicitValueScorer:
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(load_in_4bit=load_in_4bit) if is_bitsandbytes_available() else None,
        "attn_implementation": "flash_attention_2" if use_flash_attention_2 and is_flash_attn_2_available() else None,
    }
    model = AutoModelForCausalLM.from_pretrained(
        get_local_model_path("ckpt/gpt2-imdb-dpo"),
        **model_kwargs,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        get_local_model_path("lvwerra/gpt2-imdb"),
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(get_local_model_path("lvwerra/gpt2-imdb"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    implicit_value_scorer = ImplicitValueScorer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    return implicit_value_scorer


def get_dataset(dataset_name: Optional[Text] = "ZHZisZZ/imdb_preference"):
    if dataset_name == "ZHZisZZ/imdb_preference":
        dataset = load_dataset(dataset_name, split="test"
            ).rename_columns({"prompt":"raw_prompt"}).select(range(200))
    else:
        raise NotImplementedError
    return dataset
