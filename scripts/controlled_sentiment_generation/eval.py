# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 scripts/controlled_sentiment_generation/eval.py
import os
from dataclasses import dataclass
from typing import Optional, Text

import tyro
import tqdm
import torch
from datasets import Dataset, load_dataset
from transformers import pipeline


@dataclass
class ScriptArguments:
    generation_dir:  Optional[Text] = "tmp/controlled_sentiment_generation/cbs/gen"
    evaluation_dir:  Optional[Text] = "tmp/controlled_sentiment_generation/cbs/eval"


script_args = tyro.cli(ScriptArguments)
generation = load_dataset(script_args.generation_dir, split="train")

rm = pipeline(model="lvwerra/distilbert-imdb", device=0, function_to_apply="none", return_all_scores=True)
rm.tokenizer.pad_token_id = rm.model.config.eos_token_id

results = []
with torch.no_grad():
    for sample in tqdm.tqdm(generation):
        rm_output = rm(sample["prompt"] + sample["response"])[0]
        assert rm_output[1]["label"] == "POSITIVE"
        # log_p positive - log_p negative
        score = rm_output[1]["score"] - rm_output[0]["score"]
        results.append({
            "prompt": sample["prompt"],
            "response": sample["response"],
            "score": score,
        })

# raw
dataset = Dataset.from_list(results)
dataset.to_json(os.path.join(script_args.evaluation_dir, "raw.jsonl"))

# mean
scores = [result["score"] for result in results]
mean_score = sum(scores) / len(scores)
with open(os.path.join(script_args.evaluation_dir, "mean.txt"), "w") as f:
    f.write(str(mean_score))
