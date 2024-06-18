from dataclasses import dataclass
from typing import Union, List

import torch
from torch import nn
from transformers import AutoTokenizer, LlamaPreTrainedModel, LlamaModel

from src.inference_time_alignment.utils import prepare_input
from scripts.utils import get_local_model_path
from scripts.instruction_following.utils.evaluators.base import EvaluatorInput, BaseEvaluator


@dataclass
class StarlingRMEvaluator(BaseEvaluator):
    def __post_init__(self):
        class LlamaForSequenceClassification(LlamaPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.transformer = LlamaModel(config)
                self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
                self.PAD_ID = 0
                # Initialize weights and apply final processing
                self.post_init()
            
            def get_device(self):
                return self.transformer.device

            def forward(
                self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
            ):
                transformer_outputs = self.transformer(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )
                hidden_states = transformer_outputs.hidden_states[-1]
                scores = []
                rewards = self.v_head(hidden_states).squeeze(-1)
                bs = int(input_ids.shape[0])
                for i in range(bs):
                    c_inds = (input_ids[i] == self.PAD_ID).nonzero()
                    c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
                    scores.append(rewards[i, c_ind - 1])
                scores = torch.stack(scores)
                return {"scores": scores}

        self.starlingrm_template = "<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        self.reward_model = LlamaForSequenceClassification.from_pretrained(get_local_model_path("Nexusflow/Starling-RM-34B"), torch_dtype=torch.bfloat16, device_map="auto")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(get_local_model_path("01-ai/Yi-34B-Chat"))
        self.reward_tokenizer.truncation_side = "left"
        self.reward_model.eval().requires_grad_(False)

    @torch.no_grad()
    def eval(self, input: Union[EvaluatorInput, List[EvaluatorInput]]) -> List[float]:
        non_list = False
        if not isinstance(input, list): 
            input = [input]
            non_list = True
        scores = []
        for sample in input:
            sample_text = self.starlingrm_template.format(raw_prompt=sample.prompt, response=sample.response)
            sample_tokenized = prepare_input(self.reward_tokenizer(
                sample_text, 
                truncation=True,
                max_length=2048,
                padding="max_length",
                return_tensors="pt"
            ))
            reward = self.reward_model(**sample_tokenized)['scores'].item()
            scores.append(reward)
        if non_list: scores = scores[0]
        return scores


if __name__ == "__main__":
    evaluator = StarlingRMEvaluator()
    result = evaluator.eval([
        EvaluatorInput(prompt="what is your name?", response="fuck off"),
        EvaluatorInput(prompt="what is your name?", response="my name is chatgpt"),
    ])
    print(result)
