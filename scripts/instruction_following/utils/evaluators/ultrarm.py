from dataclasses import dataclass
from typing import Union, List

import torch

from src.inference_time_alignment.utils import prepare_input
from scripts.utils import get_local_model_path
from scripts.instruction_following.utils.evaluators.base import EvaluatorInput, BaseEvaluator


@dataclass
class UltraRMEvaluator(BaseEvaluator):
    def __post_init__(self):
        from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
        import torch.nn as nn
        import torch
        from typing import Optional, List

        class LlamaRewardModel(PreTrainedModel):
            config_class = LlamaConfig
            def __init__(self, config):
                super().__init__(config)
                self.model = LlamaModel(config)
                self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

            def forward( # args are the same as LlamaForCausalLM
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
            ):

                transformer_outputs = self.model(
                                        input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds,                               
                                    )

                hidden_states = transformer_outputs[0]
                rewards = self.regression_head(hidden_states).squeeze(-1)
                
                ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
                rewards = torch.gather(rewards, 1, ends)
                
                return rewards

        self.ultrarm_template = """Human: {raw_prompt}\n\nAssistant: {response}"""
        self.tokenizer = LlamaTokenizer.from_pretrained(get_local_model_path("openbmb/UltraRM-13b"))
        self.model = LlamaRewardModel.from_pretrained(get_local_model_path("openbmb/UltraRM-13b"), torch_dtype=torch.bfloat16).cuda()

    @torch.no_grad()
    def eval(self, input: Union[EvaluatorInput, List[EvaluatorInput]]) -> List[float]:
        non_list = False
        if not isinstance(input, list): 
            input = [input]
            non_list = True
        scores = []
        for sample in input:
            sample_text = self.ultrarm_template.format(raw_prompt=sample.prompt, response=sample.response)
            sample_tokenized = prepare_input(self.tokenizer(sample_text, return_tensors="pt"))
            reward = self.model(**sample_tokenized).item()
            scores.append(reward)
        if non_list: scores = scores[0]
        return scores


if __name__ == "__main__":
    evaluator = UltraRMEvaluator()
    result = evaluator.eval([
        EvaluatorInput(prompt="what is your name?", response="fuck off"),
        EvaluatorInput(prompt="what is your name?", response="my name is chatgpt"),
    ])
    print(result)
