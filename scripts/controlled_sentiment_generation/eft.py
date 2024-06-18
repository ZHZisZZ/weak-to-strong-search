import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Text

import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available
from datasets import Dataset

from src.inference_time_alignment.decoders.eft import EFTPosthocGenerationMixin
from src.inference_time_alignment.models import PrefixPreTrainedWrapper
from scripts.controlled_sentiment_generation.utils import get_scorer, get_dataset
from scripts.utils import (
    set_seeds, get_local_model_path, split_dataset_per_rank,
    get_output_path, same_tokenizer, GenConfig
)


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
@dataclass
class EFTGenConfig:
    # EFT related args
    beta: Optional[float] = 1.0  # inference-time scaling factor for EFT
    # other args
    others: GenConfig = field(default_factory=lambda: GenConfig(max_new_tokens=50, temperature=0.7, top_p=1.0))


@dataclass
class ScriptArguments:
    model_name:            Optional[Text] = "openai-community/gpt2-large"
    base_prompt_template:  Optional[str]  = "Here is a movie review from imdb: {raw_prompt}" # zero-shot prompt for base models
    dataset_name:          Optional[Text] = "ZHZisZZ/imdb_preference"
    output_dir:            Optional[Text] = "tmp/controlled_sentiment_generation/eft/gen"
    overwrite:             Optional[bool] = False
    rank:                  Optional[int]  = 1 # one-based indexing
    world_size:            Optional[int]  = 1
    seed:                  Optional[int]  = 1
    load_in_4bit:          Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    gen: EFTGenConfig = field(default_factory=lambda: EFTGenConfig()) # eft related configs


script_args = tyro.cli(ScriptArguments)
print(script_args)
set_seeds(script_args.seed)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
# load dataset
print(f"loading dataset {script_args.dataset_name} ...")
dataset = get_dataset(script_args.dataset_name)
# split dataset by rank and append rank suffix to output path, e.g., "00001-00008.jsonl"
dataset = split_dataset_per_rank(dataset, script_args.rank, script_args.world_size)
output_path = get_output_path(script_args.output_dir, script_args.rank, script_args.world_size)
# skip if previous generation result exists and we do not want to overwrite it
if os.path.exists(output_path) and not script_args.overwrite: exit()

# load base model, tokenizer and prompt template for base model
print(f"loading base model {script_args.model_name} ...")
base = AutoModelForCausalLM.from_pretrained(
    get_local_model_path(script_args.model_name),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=script_args.load_in_4bit) if is_bitsandbytes_available() else None,
    attn_implementation="flash_attention_2" if script_args.use_flash_attention_2 and is_flash_attn_2_available() else None,
)
tokenizer = AutoTokenizer.from_pretrained(get_local_model_path(script_args.model_name))

# get scorer
print("loading scorer...")
scorer = get_scorer(
    load_in_4bit=script_args.load_in_4bit,
    use_flash_attention_2=script_args.use_flash_attention_2,
)

# check that vocabulary match; note that this is just a rough sanity check; caution needed
if not same_tokenizer(tokenizer, scorer.tokenizer): 
    raise Exception("EFT requires models to share the same vocabulary")

#-----------------------------------------------------------------------------#
#---------------------------------- sample -----------------------------------#
#-----------------------------------------------------------------------------#
results = []
for raw_prompt in tqdm.tqdm(dataset["raw_prompt"]):
    eft_model  = EFTPosthocGenerationMixin(
        base   = PrefixPreTrainedWrapper(base,
                                         tokenizer,
                                         script_args.base_prompt_template.format(raw_prompt=raw_prompt)),
        tune_r = PrefixPreTrainedWrapper(scorer.model,
                                         scorer.tokenizer,
                                         scorer.model_prompt_template.format(raw_prompt=raw_prompt)),
        base_r = PrefixPreTrainedWrapper(scorer.ref_model,
                                         scorer.tokenizer,
                                         scorer.ref_model_prompt_template.format(raw_prompt=raw_prompt)),
        w      = script_args.gen.beta,
    )
    outputs = eft_model.generate(**asdict(script_args.gen.others))
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({
        "prompt": raw_prompt,
        "response": response,
    })

dataset = Dataset.from_list(results)
dataset.to_json(output_path)
