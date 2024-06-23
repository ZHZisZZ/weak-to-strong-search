# Instruction Following

This directory contains code and instructions for using off-the-shelf small/weak models to guide the decoding of large/strong models to better follow human instructions.

## Supported Models

The small/weak model pairs we currently support (in the order of tuned and untuned):
- Zephyr guidance: [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [`HuggingFaceH4/mistral-7b-sft-beta`](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta)
- Starling guidance: [`berkeley-nest/Starling-LM-7B-alpha`](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha), [`openchat/openchat_3.5`](https://huggingface.co/openchat/openchat_3.5) 
- Tulu guidance: [`allenai/tulu-2-dpo-7b`](https://huggingface.co/allenai/tulu-2-dpo-7b), [`allenai/tulu-2-7b`](https://huggingface.co/allenai/tulu-2-7b)

To add customized model pairs, see `get_scorer` function from [`scripts/instruction_following/utils/utils.py`](https://github.com/ZHZisZZ/weak-to-strong-search/blob/main/scripts/instruction_following/utils/utils.py) function.

The large/strong base models we currently support:
- Llama-2 series: [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- Llama-3 series: [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [`meta-llama/Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)

To add customized base model, modify `get_chat_prompt_template` function from [`scripts/instruction_following/utils/utils.py`](https://github.com/ZHZisZZ/weak-to-strong-search/blob/main/scripts/instruction_following/utils/utils.py) to provide chat template.


## Guided Generation

We demonstrate how to test guided models on [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). 

Here are examples of steering [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) under Zephyr guidance ([`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [`HuggingFaceH4/mistral-7b-sft-beta`](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta)) on a subset of prompts from [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) (1/32 * 805):

- CBS (Chunk-level Beam Search) with W, K, L = 2, 2, 30:

    ```bash
    PYTHONPATH=. python scripts/instruction_following/cbs.py \
        --rank=1 --world_size=32 \
        --gen.w=2 --gen.k=2 --gen.l=30 \
        --model_name="meta-llama/Llama-2-13b-chat-hf" \
        --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
        --output_dir="output/instruction_following/cbs/w2k2l30/gen"
    ```

- BoN (Best-of-N Sampling) with N = 4 or N = 8:
    ```bash
     PYTHONPATH=. python scripts/instruction_following/cbs.py \
        --rank=1 --world_size=32 \
        --gen.w=1 --gen.k=4 --gen.l=None \
        --model_name="meta-llama/Llama-2-13b-chat-hf" \
        --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
        --output_dir="output/instruction_following/bon/n4/gen"

     PYTHONPATH=. python scripts/instruction_following/cbs.py \
        --rank=1 --world_size=32 \
        --gen.w=1 --gen.k=8 --gen.l=None \
        --model_name="meta-llama/Llama-2-13b-chat-hf" \
        --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
        --output_dir="output/instruction_following/bon/n8/gen"
    ```

- EFT (Emulated Fine-Tuning):
    ```bash
    PYTHONPATH=. python scripts/instruction_following/eft.py \
        --rank=1 --world_size=32 \
        --gen.beta=1.0 \
        --model_name="meta-llama/Llama-2-13b-chat-hf" \
        --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
        --output_dir="output/instruction_following/eft/beta1.0/gen"
    ```
    *Note that EFT is only applicable when all models share the same vocabulary.*

- Base w/o guidance:
    ```bash
    PYTHONPATH=. python scripts/instruction_following/cbs.py \
        --rank=1 --world_size=32 \
        --gen.w=1 --gen.k=1 --gen.l=None \
        --model_name="meta-llama/Llama-2-13b-chat-hf" \
        --output_dir="output/instruction_following/base/gen"
    ```

(Optionally) Repeat over all ranks for complete generation results, but a subset is usually enough for a sanity check.:
```bash
for i in $(seq 1 32); do
    PYTHONPATH=. python scripts/instruction_following/cbs.py \
        --rank=${rank} --world_size=32 \
        --gen.w=2 --gen.k=2 --gen.l=30 \
        --model_name="meta-llama/Llama-2-13b-chat-hf" \
        --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
        --output_dir="output/instruction_following/cbs/w2k2l30/gen"
done
```

## Evaluation

There are three ways to automatically evaluate the generated responses: 1. GPT-4 (AlpacaEval default) 2. [`openbmb/UltraRM-13b`](openbmb/UltraRM-13b) and 3. [`Nexusflow/Starling-RM-34B`](https://huggingface.co/Nexusflow/Starling-RM-34B):

- GPT-4:

    ```bash
    PYTHONPATH=. python scripts/instruction_following/eval.py \
        --evaluator_name="GPT-4" \
        --generation_dir="output/instruction_following/cbs/w2k2l30/gen" \
        --evaluation_dir="output/instruction_following/cbs/w2k2l30/eval"

     OPENAI_API_KEY="..." alpaca_eval --model_outputs "output/instruction_following/cbs/w2k2l30/eval/GPT-4/model_outputs.json"
    ```

- [`openbmb/UltraRM-13b`](https://huggingface.co/openbmb/UltraRM-13b) and [`Nexusflow/Starling-RM-34B`](https://huggingface.co/Nexusflow/Starling-RM-34B):

    ```bash
    for evaluator_name in "openbmb/UltraRM-13b" "Nexusflow/Starling-RM-34B"; do
        PYTHONPATH=. python scripts/instruction_following/eval.py \
            --evaluator_name=${evaluator_name} \
            --generation_dir="output/instruction_following/cbs/w2k2l30/gen" \
            --evaluation_dir="output/instruction_following/cbs/w2k2l30/eval"
    done
    ```

# FQA

<details>
<summary>To decode models saved locally.</summary>

If you do not save models in the default cache directory (e.g., `~/.cache/huggingface`), modify [`scripts/configs/local_model_path.yaml`](https://github.com/ZHZisZZ/weak-to-strong-search/blob/main/scripts/configs/local_model_path.yaml) to map model name to its local path. For example.

```yaml
meta-llama/Meta-Llama-3-8B-Instruct: ~/models/Meta-Llama-3-8B-Instruct
meta-llama/Meta-Llama-3-70B-Instruct: ~/models//Meta-Llama-3-70B-Instruct
```
</details>

<details>
<summary>Out of GPU memory.</summary>

To infer a large (70B) model that doesn't fit on a single GPU, run the code as is with multiple GPUs or 4-bit quantization. For example:

```sh
# Infer on one single GPU
CUDA_VISIBLE_DEVICES=0 python ...

# Infer on one single GPU with 4-bit quant
CUDA_VISIBLE_DEVICES=0 python ... --load_in_4bit=True

# Infer on four GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python ...

# Infer on four GPUs with 4-bit quant
CUDA_VISIBLE_DEVICES=0,1,2,3 python ... --load_in_4bit=True
```
</details>