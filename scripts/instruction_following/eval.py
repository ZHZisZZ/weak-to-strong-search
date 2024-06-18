import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Text

import tyro
import tqdm
import torch
from datasets import Dataset, load_dataset

from scripts.instruction_following.utils.evaluators.base import EvaluatorInput
from scripts.instruction_following.utils import get_evaluator


@dataclass
class ScriptArguments:
    evaluator_name:  Optional[Text] = "openbmb/UltraRM-13b" # "openbmb/UltraRM-13b", "Nexusflow/Starling-RM-34B", "GPT-4"
    generation_dir:  Optional[Text] = "tmp/instruction_following/cbs/gen"
    evaluation_dir:  Optional[Text] = "tmp/instruction_following/cbs/eval"


script_args = tyro.cli(ScriptArguments)
generation = load_dataset(script_args.generation_dir, split="train")

# if "GPT-4" as evaluator, simply aggregate the generations for alpaca_eval later 
if script_args.evaluator_name == "GPT-4":

    generation = generation.map(lambda sample: {"generator": script_args.generation_dir})
    Path(Path(script_args.evaluation_dir) / script_args.evaluator_name).mkdir(parents=True, exist_ok=True)
    with open(Path(script_args.evaluation_dir) / script_args.evaluator_name / "model_outputs.json", 'w+') as file:
        json.dump(generation.to_list(), file, indent=4)

# else, use "openbmb/UltraRM-13b", ""Nexusflow/Starling-RM-34B"" for scalar preference scores.
else:

    print("loading evaluator...")
    evaluator = get_evaluator(script_args.evaluator_name)

    results = []
    with torch.no_grad():
        for sample in tqdm.tqdm(generation):
            score = evaluator.eval(EvaluatorInput(prompt=sample["instruction"], response=sample["output"]))
            results.append({
                "instruction": sample["instruction"],
                "output": sample["output"],
                "score": score,
            })

    # raw
    dataset = Dataset.from_list(results)
    dataset.to_json(Path(script_args.evaluation_dir) / script_args.evaluator_name / "raw.jsonl")

    # mean
    scores = [result["score"] for result in results]
    mean_score = sum(scores) / len(scores)
    with open(Path(script_args.evaluation_dir) / script_args.evaluator_name / "mean.txt", "w") as f:
        f.write(str(mean_score))
