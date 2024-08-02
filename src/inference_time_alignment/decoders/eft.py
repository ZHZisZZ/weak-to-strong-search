from dataclasses import dataclass
from typing import List, Any, Optional, Union

import torch
from transformers import PreTrainedModel, GenerationMixin


@dataclass(kw_only=True)
class EFTPosthocGenerationMixin(GenerationMixin):
    """
    A wrapper class that decodes base, tune_r, and base_r in parallel and combines their distributions using linear weights.
    The resulting sampling distribution is:
    softmax{ logp_{base} + sum_{i=1}^N (w_i * (logp_{tune_r[i]} - logp_{base_r})) }

    Attributes:
        base: (PreTrainedModel) -- The base language model to steer at inference time.
        tune_r: (Union[List[PreTrainedModel], PreTrainedModel]) -- The models fine-tuned from base_r.
        base_r: (Optional[PreTrainedModel], defaults to base) -- The pre-trained model for tune-r, default to base if unspecified.
        w: (Union[List[float], float]) -- The linear weights for combining the implicit reward models.
    """
    base: PreTrainedModel
    tune_r: Union[List[PreTrainedModel], PreTrainedModel]
    base_r: Optional[PreTrainedModel] = None
    w: Union[List[float], float]

    def __post_init__(self):
        if not isinstance(self.base, list):  self.tune_r = [self.tune_r]
        if not isinstance(self.w, list): self.w = [self.w]
        if not self.base_r: self.base_r = self.base
        assert len(self.tune_r) == len(self.w) 

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        if 'past_key_values' in model_kwargs:
            past_key_values = model_kwargs['past_key_values']
            model_kwargs['past_key_values'] = model_kwargs['past_key_values']['base']
        result = self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if 'past_key_values' in model_kwargs:
            result['past_key_values'] = past_key_values
        return result

    def _parallel_decode(self, *args, past_key_values_list, **kwargs):
        models_outputs_list = []
        for i, model in enumerate(self.tune_r):
            outputs = model(*args, past_key_values=past_key_values_list[i], **kwargs)
            models_outputs_list.append(outputs)
        return models_outputs_list

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values:
            past_key_values = {'base': None, 'tune_r': [None] * len(self.tune_r), 'base_r': None}

        base_outputs = self.base(*args, past_key_values=past_key_values['base'], **kwargs)
        base_logits  = base_outputs.logits[:, -1, :]

        if self.base == self.base_r:
            base_r_outputs = base_outputs
            base_r_logits  = base_logits
        else:
            base_r_outputs = self.base_r(*args, past_key_values=past_key_values['base_r'], **kwargs)
            base_r_logits  = base_r_outputs.logits[:, -1, :]

        tune_r_outputs_list = self._parallel_decode(*args, past_key_values_list=past_key_values['tune_r'], **kwargs)
        tune_r_logits_list  = [outputs.logits[:, -1, :] for outputs in tune_r_outputs_list]

        tune_r_logits = torch.stack(tune_r_logits_list, dim=-1)
        w = torch.tensor(self.w).to(tune_r_logits.device)
        r = (w * (tune_r_logits - base_r_logits.unsqueeze(-1))).sum(-1)

        logits = base_logits + r

        outputs = base_outputs
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = {
            'base':   base_outputs.past_key_values,
            'tune_r': [outputs.past_key_values for outputs in tune_r_outputs_list],
            'base_r': base_r_outputs.past_key_values,
        }
        return outputs


if __name__ == "__main__":
    # unit test
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.inference_time_alignment.models import PrefixPreTrainedWrapper

    # Load the tokenizer and model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix_a = "This is a world governed by angry bananas. "
    prefix_b = "This is a world governed by zhanhui zhou. "
    # Define some example input texts
    input_texts = [
        "Once upon a time in a faraway land, ",
        "In a galaxy far, ",
        "In the beginning, "
        "12345"
    ]
    generate_kwargs = {
        "max_new_tokens": 50,
        "do_sample": False,
    }

    inputs = tokenizer([prefix_a + input_text for input_text in input_texts], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs,
        )
    outputs_a = outputs[:, -generate_kwargs["max_new_tokens"]:]

    inputs = tokenizer([prefix_b + input_text for input_text in input_texts], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs,
        )
    outputs_b = outputs[:, -generate_kwargs["max_new_tokens"]:]


    eft_model = EFTPosthocGenerationMixin(
        base   = PrefixPreTrainedWrapper(model=model, tokenizer=tokenizer, prefix=prefix_a, add_special_tokens=True),
        tune_r = PrefixPreTrainedWrapper(model=model, tokenizer=tokenizer, prefix=prefix_b, add_special_tokens=True),
        w = 0
    )
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = eft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs
        )
    outputs_a_ = outputs[:, -generate_kwargs["max_new_tokens"]:]

    eft_model = EFTPosthocGenerationMixin(
        base   = PrefixPreTrainedWrapper(model=model, tokenizer=tokenizer, prefix=prefix_a, add_special_tokens=True),
        tune_r = PrefixPreTrainedWrapper(model=model, tokenizer=tokenizer, prefix=prefix_b, add_special_tokens=True),
        w = 1
    )
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = eft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs
        )
    outputs_b_ = outputs[:, -generate_kwargs["max_new_tokens"]:]

    assert (outputs_a_ == outputs_a).min().item() == True
    assert (outputs_b_ == outputs_b).min().item() == True
