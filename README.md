<!-- TODO: change `weak-to-strong-search` ot `weak-to-strong-search` -->

# Weak-to-Strong Search

Code release for [Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models](https://arxiv.org/abs/2405.19262).

- The [`scripts/instruction_following`](https://github.com/ZHZisZZ/weak-to-strong-search/blob/main/scripts/instruction_following) directory contains code and instructions for using off-the-shelf small/weak models to guide the decoding of large/strong models to better follow human instructions.

- The [`scripts/controlled_sentiment_generation`](https://github.com/ZHZisZZ/weak-to-strong-search/blob/main/scripts/controlled_sentiment_generation) directory contains code and instructions for using tuned and untuned gpt2s (124M) to control larger models to write positive movie reviews.


## Installation

```bash
conda create -n weak-to-strong-search python=3.10
conda activate weak-to-strong-search
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# (optional) pip install flash-attn==2.3.2 --no-build-isolation
# (optional) pip install bitsandbytes==0.42.0
```

## Quick Start

<details>
<summary>
(Click to expand) To use <a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta"><code>HuggingFaceH4/zephyr-7b-beta</code></a> and its untuned verision <a href="https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta"><code>HuggingFaceH4/mistral-7b-sft-beta</code></a> to guide the decoding of <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"><code>meta-llama/Meta-Llama-3-8B-Instruct</code></a> for better alignment.
</summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference_time_alignment.decoders.cbs import CBSPosthocGenerationMixin
from src.inference_time_alignment.scorers import ImplicitValueScorer


def get_zephyr_scorer() -> ImplicitValueScorer:
    """
    Use `zephyr-7b-beta` and its untuned verision `mistral-7b-sft-beta` as scorer to guide other models
    """
    tuned_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
    untuned_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/mistral-7b-sft-beta", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompt_template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user",   "content": "{raw_prompt}"},
        ],
        tokenize=False, 
        add_generation_prompt=True,
    )
    implicit_value_scorer = ImplicitValueScorer(
        model=tuned_model,
        ref_model=untuned_model,
        tokenizer=tokenizer,
        model_prompt_template=prompt_template,
        ref_model_prompt_template=prompt_template,
    )
    return implicit_value_scorer


# the (stonger/larger) model to be guided
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt_template = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": ""},
        {"role": "user",   "content": "{raw_prompt}"},
    ],
    tokenize=False, 
    add_generation_prompt=True,
)

# chunk-level beam search wrapper
cbs_model = CBSPosthocGenerationMixin(base, tokenizer)
# implicit value scorer
scorer = get_zephyr_scorer()

# prepare prompts
raw_prompt = "Who are you?"
prompt = prompt_template.format(raw_prompt=raw_prompt)
prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
prompt_len = prompt_tokenized["input_ids"].size(1)

# search for the highest scoring response
outputs = cbs_model.search(
    input_ids=prompt_tokenized["input_ids"].cuda(),
    attention_mask=prompt_tokenized["attention_mask"].cuda(),
    scorer=scorer.set_raw_prompt(raw_prompt),
    split_by_prompt_text=False,
    w=2, k=2, l=30, # CBS related args 
    max_new_tokens=128,
)

print(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True))
```

</details>


See [`scripts/instruction_following`](https://github.com/ZHZisZZ/weak-to-strong-search/blob/main/scripts/instruction_following) for more examples.


## Reference

```
@article{zhou2024weak,
  title={Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models},
  author={Zhou, Zhanhui and Liu, Zhixuan and Liu, Jie and Dong, Zhichen and Yang, Chao and Qiao, Yu},
  journal={arXiv preprint arXiv:2405.19262},
  year={2024}
}
```
