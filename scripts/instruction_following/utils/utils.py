from typing import Optional, Text

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available

from src.inference_time_alignment.scorers import ImplicitValueScorer
from scripts.utils import get_local_model_path


def get_chat_prompt_template(model_name: Text, tokenizer: PreTrainedTokenizer) -> Text:
    if model_name in ("meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        ) + " " # add a trailing space
    elif model_name in ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/mistral-7b-sft-beta"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("berkeley-nest/Starling-LM-7B-alpha", "openchat/openchat_3.5"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        ) + " " # add a trailing space
    elif model_name in ("allenai/tulu-2-dpo-7b", "allenai/tulu-2-7b"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        )
    # modify here to support your customized models
    else:
        raise NotImplementedError


def get_scorer(
    scorer_name, 
    load_in_4bit: Optional[bool] = False, 
    use_flash_attention_2: Optional[bool] = False,
) -> ImplicitValueScorer:
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit and is_bitsandbytes_available() else None,
        "attn_implementation": "flash_attention_2" if use_flash_attention_2 and is_flash_attn_2_available() else None,
    }

    # map score_name to (tuned name and untuned name)
    scorer_map = {
        "HuggingFaceH4/zephyr-7b-beta": ("HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/mistral-7b-sft-beta"),
        "berkeley-nest/Starling-LM-7B-alpha": ("berkeley-nest/Starling-LM-7B-alpha", "openchat/openchat_3.5"),
        "allenai/tulu-2-dpo-7b": ("allenai/tulu-2-dpo-7b", "allenai/tulu-2-7b"),
        # modify here to support your customized models
    }

    if scorer_name not in scorer_map: 
        raise NotImplementedError(f"{scorer_name} is not supported")
    
    tuned_name = scorer_map[scorer_name][0]
    untuned_name = scorer_map[scorer_name][1]

    tuned_model = AutoModelForCausalLM.from_pretrained(get_local_model_path(tuned_name), **model_kwargs)
    untuned_model = AutoModelForCausalLM.from_pretrained(get_local_model_path(untuned_name), **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(get_local_model_path(tuned_name))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    implicit_value_scorer = ImplicitValueScorer(
        model=tuned_model,
        ref_model=untuned_model,
        tokenizer=tokenizer,
        model_prompt_template=get_chat_prompt_template(tuned_name, tokenizer),
        ref_model_prompt_template=get_chat_prompt_template(untuned_name, tokenizer),
    )
    return implicit_value_scorer


def get_evaluator(evaluator_name: Text):
    from scripts.instruction_following.utils.evaluators.starlingrm import StarlingRMEvaluator
    from scripts.instruction_following.utils.evaluators.ultrarm import UltraRMEvaluator
    if evaluator_name == "Nexusflow/Starling-RM-34B":
        return StarlingRMEvaluator()
    elif evaluator_name == "openbmb/UltraRM-13b":
        return UltraRMEvaluator()
    else:
        raise NotImplementedError


def get_dataset(dataset_name: Optional[Text] = "tatsu-lab/alpaca_eval"):
    if dataset_name == "tatsu-lab/alpaca_eval":
        dataset = load_dataset(dataset_name, split="eval").rename_columns({"instruction":"raw_prompt"})
    else:
        raise NotImplementedError
    return dataset
