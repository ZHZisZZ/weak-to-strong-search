from dataclasses import dataclass, asdict
from typing import Text, List, Dict, Optional
from abc import ABC, abstractclassmethod

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from src.inference_time_alignment.utils import (
    SFTDataMapFunc, 
    SFTDataCollatorWithPadding,
    get_batch_logps,
    prepare_input
)


DEFAULT_PROMPT_TEMPLATE = "{raw_prompt}"


@dataclass
class ScorerInput:
    response: List[str]
    eos: List[bool]


@dataclass
class BaseScorer(ABC):
    
    @abstractclassmethod
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class ImplicitValueScorer(BaseScorer):
    model: PreTrainedModel
    ref_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = False
    model_prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    ref_model_prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    raw_prompt: Optional[str] = None

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        policy_all_logps = self.forward(
            self.model, 
            self.model_prompt_template, 
            input
        )
        ref_all_logps = self.forward(
            self.ref_model, 
            self.ref_model_prompt_template, 
            input
        )
        return policy_all_logps - ref_all_logps

    @torch.no_grad()
    def forward(
        self, 
        model: PreTrainedModel, 
        prompt_template: Text, 
        input: ScorerInput | Dict
    ) -> torch.Tensor:
        input = asdict(input)
        prompt = prompt_template.format(raw_prompt=self.raw_prompt)
        input["prompt"] = [prompt] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, 
                                add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))])
        batch = prepare_input(batch)

        all_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.to(torch.float32)

        return get_batch_logps(all_logits, batch["labels"])


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator

    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/hwfile/llm-safety/models/gpt2-imdb-dpo",
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        "/mnt/hwfile/llm-safety/models/gpt2-imdb",
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    tokenizer = AutoTokenizer.from_pretrained("/mnt/hwfile/llm-safety/models/gpt2-imdb-dpo")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    implicit_reward = ImplicitValueScorer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    implicit_reward.set_raw_prompt("I think this movie is ")

    result = implicit_reward(
        ScorerInput(response=[" interesting", " boring"], eos=[True, True])
    )

    print(result)
