import os
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Text 

import yaml
from transformers import PreTrainedTokenizer
from datasets import Dataset


INT_INFINITY = 2**63 - 1


@dataclass
class GenConfig:
    do_sample:      Optional[bool]  = True
    max_new_tokens: Optional[int]   = 512
    temperature:    Optional[float] = 0.7
    top_p:          Optional[float] = 1.0


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_local_model_path(model_name) -> Text:
    # inspect `scripts/local_model_path.yaml` to optionally use local model checkpoints
    if not (Path(__file__).parent.parent / "scripts/configs/local_model_path.yaml").exists(): return model_name
    with open(Path(__file__).parent.parent / "scripts/configs/local_model_path.yaml", "r") as stream:
        model_name_to_local_path = yaml.safe_load(stream)
    return model_name_to_local_path.get(model_name, model_name)


def split_dataset_per_rank(dataset: Dataset, rank: int, world_size: int) -> Text:
    assert 1 <= rank <= world_size
    split_size = math.ceil(len(dataset) / world_size)
    dataset = dataset.select(range(
        (rank-1)*split_size, 
        min((rank)*split_size, len(dataset))
    ))
    return dataset


def get_output_path(output_dir: str, rank: int, world_size: int, suffix: str = "jsonl") -> Text:
    assert 1 <= rank <= world_size
    return os.path.join(output_dir, f"{str(rank).zfill(5)}-of-{str(world_size).zfill(5)}.{suffix}")


def same_tokenizer(tokenizer_a: PreTrainedTokenizer, tokenizer_b: PreTrainedTokenizer) -> bool:
    """Note that this is a rough check; caution needed"""
    # \blindtext from LaTex
    check_paragraph = ("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod "
                       "tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. "
                       "At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, "
                       "no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, "
                       "consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et "
                       "dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo "
                       "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.")
    if tokenizer_a.vocab_size != tokenizer_b.vocab_size: return False
    if tokenizer_a.encode(check_paragraph) != tokenizer_a.encode(check_paragraph): return False
    return True
