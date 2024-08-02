from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationMixin


@dataclass
class PrefixPreTrainedWrapper(GenerationMixin):
    """
    Wrap an auto-regressive language model so that generations will be concatenated to the prefix string. 
    This is useful when multiple language models are decoded at the same time, but not sharing the same template or context. 
    In this case, please use the wrapper together with `PosthocGenerationMixin`.

    Attributes:
        model: (`PreTrainedModel`) -- The model for generation.
        tokenizer: (`PreTrainedTokenizer`) -- The tokenizer to use.
        prefix: (`str`) -- The prefix string for generation.
        add_special_tokens: (`bool`) -- Whether to add eos_token when tokenizing.

    Example: 
        PrefixPreTrainedWrapper(
            model, 
            tokenizer,
            prefix="1234567"
        ).generate() -> "891011...."
    """
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    prefix: str
    add_special_tokens: Optional[bool] = False

    def __post_init__(self):
        self.tokenizer = deepcopy(self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.input_ids = None
        self.attention_mask = None
        self.cache_position = None

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    def _concat_prefix_to_batch(self, input_ids):
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prefix_inputs = [self.prefix + input for input in inputs]
        batch = self.tokenizer(
            text=prefix_inputs,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        ).to(input_ids.device)
        return batch["input_ids"], batch["attention_mask"]

    def __call__(
        self, 
        input_ids, 
        attention_mask, 
        use_cache=False, 
        past_key_values=None, 
        cache_position=None, 
        position_ids=None, 
        **kwargs
    ):
        del attention_mask, cache_position, position_ids
        if not past_key_values:
            self.input_ids, self.attention_mask = self._concat_prefix_to_batch(input_ids)
            self.cache_position = self.model._get_initial_cache_position(
                self.input_ids, 
                {"attention_mask": self.attention_mask, "use_cache": use_cache}
            )["cache_position"]
        else:
            self.input_ids = torch.cat([self.input_ids, input_ids[:, -1, None]], dim=-1)
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=self.input_ids, 
            attention_mask=self.attention_mask,
            use_cache=use_cache, 
            past_key_values=past_key_values, 
            cache_position=self.cache_position
        )
        model_outputs = self.model(**model_inputs)
        # _update_model_kwargs_for_generation
        self.attention_mask = torch.cat([
            self.attention_mask, self.attention_mask.new_ones((self.attention_mask.shape[0], 1))], dim=-1)
        self.cache_position = self.cache_position[-1:] + 1
        return model_outputs


if __name__ == "__main__":
    # unit test
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load the tokenizer and model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix = "This is a world governed by angry bananas. "
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

    inputs = tokenizer([prefix + input_text for input_text in input_texts], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs,
        )
    outputs_a = outputs[:, -generate_kwargs["max_new_tokens"]:]


    prefix_model = PrefixPreTrainedWrapper(model=model, tokenizer=tokenizer, prefix=prefix, add_special_tokens=True)
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = prefix_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generate_kwargs
        )
    outputs_b = outputs[:, -generate_kwargs["max_new_tokens"]:]
    assert (outputs_a == outputs_b).min().item() == True
