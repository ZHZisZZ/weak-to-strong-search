from dataclasses import dataclass
from typing import Dict, List, Text, Mapping, Optional, Any

import torch
import numpy as np
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer, StoppingCriteria


def prepare_input(data):
    # adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2626
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": Accelerator().device}
        return data.to(**kwargs)
    return data


def same_tokenizer(tokenizer_a: PreTrainedTokenizer, tokenizer_b: PreTrainedTokenizer) -> bool:
    """Note that this is a rough check; caution needed"""
    # \blindtext from LaTex
    check_paragraph = ("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod "
                       "tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. "
                       "At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, "
                       "no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, "
                       "consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et "
                       "dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo "
                       "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.")
    if tokenizer_a.vocab_size != tokenizer_b.vocab_size: return False
    if tokenizer_a.encode(check_paragraph) != tokenizer_a.encode(check_paragraph): return False
    return True


def common_prefix_length(list_a, list_b):
    length = 0
    for i in range(min(len(list_a), len(list_b))):
        if list_a[i] == list_b[i]:
            length += 1
        else:
            break
    return length


def pad_labels(features, tokenizer, pad_to_multiple_of=None, label_pad_token_id=-100):
    # copied from https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/data/data_collator.py#L562-L584
    labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.
    if labels is not None:
        max_label_length = max(len(l) for l in labels)
        if pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        padding_side = tokenizer.padding_side
        for feature in features:
            remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


@dataclass
class SFTDataMapFunc:
    """Map raw texts to tokens, attention masks, and labels."""
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    completion_only: Optional[bool] = True
    add_special_tokens: Optional[bool] = False

    def __call__(self, examples):
        new_examples = {
            "prompt_response_input_ids": [],
            "prompt_response_attention_mask": [],
            "prompt_response_labels": [],

            "prompt_input_ids": [],
            "prompt_attention_mask": [],

            "prompt": [],
        }
        for prompt, response, eos in zip(examples["prompt"], examples["response"], examples["eos"]):
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=self.add_special_tokens)
            prompt_response_tokens = self.tokenizer(prompt + response, add_special_tokens=self.add_special_tokens)
            # add EOS to response
            if eos:
                prompt_response_tokens["input_ids"].append(self.tokenizer.eos_token_id)
                prompt_response_tokens["attention_mask"].append(1)

            prompt_len = common_prefix_length(prompt_tokens["input_ids"], prompt_response_tokens["input_ids"])

            for k, toks in {
                "prompt": prompt_tokens,
                "prompt_response": prompt_response_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    new_examples[f"{k}_{type_key}"].append(tokens)
            
            for k, toks in {
                "prompt_response": prompt_response_tokens,
            }.items():
                labels = toks["input_ids"].copy()
                if self.completion_only:
                    labels[:prompt_len] = [self.label_pad_token_id] * prompt_len
                new_examples[f"{k}_labels"].append(labels) 

        new_examples["prompt"] = examples["prompt"]

        return new_examples


@dataclass
class SFTDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        return:
        batch = {
            "input_ids": ...,
            "attention_mask": ...,
            "labels": ...,
        }
        """
        right_padding_features = []
        for feature in features:
            right_padding_features.append(
                {
                    "input_ids": feature["prompt_response_input_ids"],
                    "attention_mask": feature["prompt_response_attention_mask"],
                    "labels": feature["prompt_response_labels"],
                }
            )

        pad_labels(right_padding_features, self.tokenizer, self.pad_to_multiple_of, self.label_pad_token_id)

        right_padding_batch = self.tokenizer.pad(
            right_padding_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return right_padding_batch

@dataclass
class StopOnStringCriteria(StoppingCriteria):
    start_length: int
    eos_string: Text
    tokenizer: PreTrainedTokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        return all(self.eos_string in decoded_generation for decoded_generation in decoded_generations) # Stop when ALL sequences hit the stopping critera


def extract_responses(input_ids, tokenizer, prompt: Text = None, prompt_len: int = None):
    assert not (prompt != None and prompt_len != None)
    if prompt:
        prompts_responses = tokenizer.batch_decode(input_ids)
        responses = [prompt_response[len(prompt):] for prompt_response in prompts_responses] # remove prompt
        responses = [tokenizer.decode(tokenizer(response)['input_ids'], skip_special_tokens=True) for response in responses] # remove special tokens
        return responses
    else:
        input_ids = input_ids[:, prompt_len:]
        responses = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return responses


def get_truncated_responses(responses: List[Text], eos_strings: List[Text], keep_eos_strings: bool = True):
    unfinished_sequences = torch.tensor([
        not any(eos_string in response for eos_string in eos_strings) for response in responses], 
        dtype=torch.int16,
    ).cuda()
    truncated_responses = []
    for response in responses:
        for eos_string in eos_strings:
            # sub_responses = response.split(eos_string)[0]
            # if keep_eos_strings:
            #     response += eos_string
            idx = response.find(eos_string)
            if idx != -1:
                if keep_eos_strings:
                    response = response[:idx+len(eos_string)]
                else:
                    response = response[:idx]
        truncated_responses.append(response)
    responses = truncated_responses
    return responses, unfinished_sequences
