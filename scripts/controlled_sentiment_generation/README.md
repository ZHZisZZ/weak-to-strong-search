# Controlled-Sentiment Generation

This directory contains code and instructions for using tuned and untuned gpt2s (124M param) to control larger models to write positive movie reviews.

# Setup
We have created a synthetic preference dataset of movie reviews, [`ZHZisZZ/imdb_preference`]([DPO](https://arxiv.org/abs/2305.18290)). Prompts $x$ are truncated [`imdb`](stanfordnlp/imdb) movie review (e.g., "I really") and responses $y_1, y_2$ are sampled continuations (e.g., "like ...", "hate ...") from [`lvwerra/gpt2-imdb`](https://huggingface.co/lvwerra/gpt2-imdb). We use the logit difference between positive and negative labels from [`lvwerra/distilbert-imdb`](https://huggingface.co/datasets/ZHZisZZ/imdb_preference) as the gold reward model to encourage positive continuations: $r_{\text{gold}}(x, y)= \log p(\text{positive} \mid x, y) - \log p(\text{negative} \mid x, y)$. Preferences labels are collected from this gold reward model assuming the BT preference distribution: $p(y_1 \succ y_2 \mid x) = \sigma (r_{\text{gold}}(x, y_1) - r_{\text{gold}}(x, y_2))$.


# Tune a Positive `gpt2` Model
To obtain a positive `gpt2` model, we can use [DPO](https://arxiv.org/abs/2305.18290) to tune [`lvwerra/gpt2-imdb`](https://huggingface.co/lvwerra/gpt2-imdb) for positive continuation:

```bash
python scripts/controlled_sentiment_generation/dpo.py \
    --dataset_name="ZHZisZZ/imdb_preference" \
    --model_name_or_path="lvwerra/gpt2-imdb" \
    --dataset_test_split="validation" \
    --output_dir="ckpt/gpt2-imdb-dpo" \
    --num_train_epochs=3 \
    --learning_rate=5e-4 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --weight_decay=0.05 \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=0.25 \
    --warmup_ratio=0.25 \
    --report_to wandb \
    --logging_first_step \
    --load_best_model_at_end
```

<details>
<summary>Click here for the command to train on multiple GPUs (e.g., 4).</summary>

```bash
accelerate launch --config_file scripts/configs/accelerate_configs/multi_gpu.yaml --num_processes=4 \
    scripts/controlled_sentiment_generation/dpo.py \
    --dataset_name="ZHZisZZ/imdb_preference" \
    --model_name_or_path="lvwerra/gpt2-imdb" \
    --dataset_test_split="validation" \
    --output_dir="ckpt/gpt2-imdb-dpo" \
    --num_train_epochs=3 \
    --learning_rate=5e-4 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --weight_decay=0.05 \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=0.25 \
    --warmup_ratio=0.25 \
    --report_to wandb \
    --logging_first_step \
    --load_best_model_at_end
```

</details>

The trained models will be saved to `ckpt/gpt2-imdb-dpo`. We refer to this tuned model as tuned gpt2 while the reference model [`lvwerra/gpt2-imdb`](https://huggingface.co/lvwerra/gpt2-imdb) as untuned gpt2.

# Guided Generation

Then, we demonstrate how to use tuned and untuned gpt2s (`ckpt/gpt2-imdb-dpo`, [`lvwerra/gpt2-imdb`](https://huggingface.co/lvwerra/gpt2-imdb)) (124M param) to guide a larger model from the GPT-2 family, [openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) (1.5B params), in writing positive movie reviews. The guided model continues the truncated reviews from the ([`ZHZisZZ/imdb_preference`]([DPO](https://arxiv.org/abs/2305.18290))) test split. You can use any pre-trained language model supported by [`transformers`](https://github.com/huggingface/transformers), e.g., `--model_name=meta-llama/Meta-Llama-3-8B-Instruct`.

- CBS (Chunk-level Beam Search) with W, K, L = 4, 4, 5:

    ```bash
    PYTHONPATH=. python scripts/controlled_sentiment_generation/cbs.py \
        --gen.w=4 --gen.k=4 --gen.l=5 \
        --model_name="openai-community/gpt2-xl" \
        --output_dir="output/controlled_sentiment_generation/cbs/w4k4l5/gen"
    ```

- BoN (Best-of-N Sampling) with N = 16 or N = 32:
    ```bash
     PYTHONPATH=. python scripts/controlled_sentiment_generation/cbs.py \
        --gen.w=1 --gen.k=16 --gen.l=None \
        --model_name="openai-community/gpt2-xl" \
        --output_dir="output/controlled_sentiment_generation/bon/n16/gen"

     PYTHONPATH=. python scripts/controlled_sentiment_generation/cbs.py \
        --gen.w=1 --gen.k=32 --gen.l=None \
        --model_name="openai-community/gpt2-xl" \
        --output_dir="output/controlled_sentiment_generation/bon/n32/gen"
    ```

- EFT (Emulated Fine-Tuning):
    ```bash
    PYTHONPATH=. python scripts/controlled_sentiment_generation/eft.py \
        --gen.beta=1.0 \
        --model_name="openai-community/gpt2-xl" \
        --output_dir="output/controlled_sentiment_generation/eft/beta1.0/gen"
    ```
    *Note that EFT is only applicable when all models share the same vocabulary.*

- Base w/o guidance:
    ```bash
    PYTHONPATH=. python scripts/controlled_sentiment_generation/cbs.py \
        --gen.w=1 --gen.k=1 --gen.l=None \
        --model_name="openai-community/gpt2-xl" \
        --output_dir="output/controlled_sentiment_generation/base/gen"
    ```


# Evaluation

To evaluate the generated samples with the gold reward model $r_{\text{gold}}$, run:

```bash
PYTHONPATH=. python scripts/controlled_sentiment_generation/eval.py \
    --generation_dir="output/controlled_sentiment_generation/cbs/w4k4l5/gen" \
    --evaluation_dir="output/controlled_sentiment_generation/cbs/w4k4l5/eval"
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
