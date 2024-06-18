# adapted from: https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py
"""
# Train gpt-2 on single GPU (~15 minutes on A100)
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

# Train gpt-2 on multiple (e.g., four) GPUs with DDP (~4 minutes on A100)
accelerate launch --config_file scripts/configs/accelerate_configs/multi_gpu.yaml --num_processes=4 \
    scripts/controlled_sentiment_generation/dpo.py \
    ...
"""

import multiprocessing

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
)
from trl.commands.cli_utils import DPOScriptArguments, TrlParser


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        chosen_idx = row["chosen"] # either 0 or 1
        row["chosen"]   = row["responses"][chosen_idx]
        row["rejected"] = row["responses"][1-chosen_idx]
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
