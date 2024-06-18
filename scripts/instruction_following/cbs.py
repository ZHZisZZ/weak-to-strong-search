import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Text

import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available
from datasets import Dataset

from src.inference_time_alignment.decoders.cbs import CBSPosthocGenerationMixin
from src.inference_time_alignment.utils import extract_responses
from scripts.instruction_following.utils import get_chat_prompt_template, get_scorer, get_dataset 
from scripts.utils import (
    set_seeds, get_local_model_path, split_dataset_per_rank,
    get_output_path, INT_INFINITY, GenConfig
)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
@dataclass
class CBSGenConfig:
    # CBS related args (default to disable CBS)
    w: Optional[int] = 2  # n. hypotheses to keep (beam width)
    k: Optional[int] = 2  # n. successors per hypothethis
    l: Optional[int] = 10 # chunk length
    # other args
    others: GenConfig = field(default_factory=lambda: GenConfig(max_new_tokens=2048, temperature=0.7, top_p=1.0))

    def __post_init__(self):
        if self.l == None: self.l = INT_INFINITY


@dataclass
class ScriptArguments:
    model_name:            Optional[Text] = "meta-llama/Meta-Llama-3-8B-Instruct"
    scorer_name:           Optional[Text] = "HuggingFaceH4/zephyr-7b-beta"
    dataset_name:          Optional[Text] = "tatsu-lab/alpaca_eval"
    output_dir:            Optional[Text] = "tmp/instruction_following/cbs/gen"
    overwrite:             Optional[bool] = False
    rank:                  Optional[int]  = 1 # one-based indexing
    world_size:            Optional[int]  = 1
    seed:                  Optional[int]  = 1
    load_in_4bit:          Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    gen:  CBSGenConfig = field(default_factory=lambda: CBSGenConfig()) # cbs related configs


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
    quantization_config=BitsAndBytesConfig(load_in_4bit=True) if script_args.load_in_4bit and is_bitsandbytes_available() else None,
    attn_implementation="flash_attention_2" if script_args.use_flash_attention_2 and is_flash_attn_2_available() else None,
)
tokenizer = AutoTokenizer.from_pretrained(get_local_model_path(script_args.model_name))
prompt_template = get_chat_prompt_template(script_args.model_name, tokenizer)

# get cbs model
cbs_model = CBSPosthocGenerationMixin(base, tokenizer)
# get scorer
print(f"loading scorer {script_args.scorer_name} ...")
scorer = get_scorer(
    scorer_name=script_args.scorer_name,
    load_in_4bit=script_args.load_in_4bit,
    use_flash_attention_2=script_args.use_flash_attention_2,
)

#-----------------------------------------------------------------------------#
#---------------------------------- search -----------------------------------#
#-----------------------------------------------------------------------------#
results = []
for raw_prompt, ds_id in tqdm.tqdm(zip(dataset["raw_prompt"], dataset["dataset"])):
    prompt = prompt_template.format(raw_prompt=raw_prompt)
    prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = cbs_model.search(
        input_ids=prompt_tokenized["input_ids"].cuda(),
        attention_mask=prompt_tokenized["attention_mask"].cuda(),
        scorer=scorer.set_raw_prompt(raw_prompt),
        split_by_prompt_text=False,
        w=script_args.gen.w,
        k=script_args.gen.k,
        l=script_args.gen.l, 
        **asdict(script_args.gen.others),
    )
    response = extract_responses(outputs, tokenizer, prompt_len=prompt_tokenized["input_ids"].size(1))[0]
    results.append({
        "instruction": raw_prompt,
        "output": response,
        "generator": f"{script_args.model_name}({str(script_args)})",
        "dataset": ds_id,
        "datasplit": "eval"
    })

dataset = Dataset.from_list(results)
dataset.to_json(output_path)
