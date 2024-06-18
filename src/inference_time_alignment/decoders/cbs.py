from dataclasses import dataclass
from typing import Dict, Any, Text, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationMixin
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import MaxLengthCriteria, EosTokenCriteria, StoppingCriteriaList

from src.inference_time_alignment.scorers import BaseScorer, ScorerInput
from src.inference_time_alignment.utils import StopOnStringCriteria, extract_responses, get_truncated_responses


@dataclass
class CBSPosthocGenerationMixin(GenerationMixin):
    """
    CBS: Chunk-level Beam Search (https://arxiv.org/pdf/2405.19262).
    """
    base: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __getattribute__(self, name: Text) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.base._reorder_cache(past_key_values, beam_idx)

    @torch.no_grad()
    def search(
        self,
        # key args
        input_ids: torch.LongTensor,
        scorer: BaseScorer,
        w: Optional[int] = 4,
        k: Optional[int] = 4,
        l: Optional[int] = 10,
        # other args
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_strings: Optional[int] = None,
        split_by_prompt_text: Optional[bool] = True,
        **kwargs,
    ):
        if not self.generation_config.pad_token_id:
            self.generation_config.pad_token_id = self.generation_config.eos_token_id
            if isinstance(self.generation_config.pad_token_id, list):
                self.generation_config.pad_token_id = self.generation_config.pad_token_id[0]

        # init logits_warper
        logits_warper = LogitsProcessorList()
        if temperature: logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_k: logits_warper.append(TopKLogitsWarper(top_k))
        if top_p: logits_warper.append(TopPLogitsWarper(top_p))

        # init stopping criteria
        stopping_criteria = StoppingCriteriaList()
        if eos_strings: 
            stopping_criteria.extend([StopOnStringCriteria(input_ids.size(1), eos_string, self.tokenizer) for eos_string in eos_strings])
        assert not (max_new_tokens and max_length)
        if max_length: 
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_new_tokens: 
            stopping_criteria.append(MaxLengthCriteria(max_length=input_ids.size(1)+max_new_tokens))
        if self.generation_config.eos_token_id is not None: 
            stopping_criteria.append(EosTokenCriteria(eos_token_id=self.generation_config.eos_token_id))

        return self._search(
            input_ids,
            scorer,
            w=w,
            k=k,
            l=l,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            eos_strings=eos_strings,
            split_by_prompt_text=split_by_prompt_text,
            **kwargs
        )

    @torch.no_grad()
    def _search(
        self,
        # key args
        input_ids: torch.LongTensor,
        scorer: BaseScorer,
        w: Optional[int] = 4,
        k: Optional[int] = 4,
        l: Optional[int] = 10,
        # other args
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        pad_token_id: Optional[int] = None,
        return_dict_in_generate: Optional[bool] = None,
        eos_strings: Optional[int] = None,
        split_by_prompt_text: Optional[bool] = True,
        **model_kwargs,
    ) -> Union[Dict, torch.LongTensor]:
        #-----------------------------------------------------------------------------#
        #----------------------------------- setup -----------------------------------#
        #-----------------------------------------------------------------------------#
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        # repeat input_ids and attention_mask
        input_ids = input_ids.repeat(w * k, 1)
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat(w * k, 1)

        # keep track of which sequences are already finished
        this_peer_finished = False  # used by synced_gpus only
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # cbs related params
        tokens_remain_per_chunk = l
        prompt, prompt_len = self.tokenizer.decode(input_ids[0]), input_ids.size(1)

        while not this_peer_finished:
            #-----------------------------------------------------------------------------#
            #------------------------------ regular decoding -----------------------------#
            #-----------------------------------------------------------------------------#
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.__call__(
                **model_inputs,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = logits_processor(input_ids, next_token_logits)
            next_token_logits = logits_warper(input_ids, next_token_logits)
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            tokens_remain_per_chunk -= 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

            #-----------------------------------------------------------------------------#
            #------------------------------ state selection ------------------------------#
            #-----------------------------------------------------------------------------#
            if tokens_remain_per_chunk <= 0 or this_peer_finished == True:
                tokens_remain_per_chunk = l

                if split_by_prompt_text:
                    responses = extract_responses(input_ids, self.tokenizer, prompt=prompt)
                else:
                    responses = extract_responses(input_ids, self.tokenizer, prompt_len=prompt_len)
                if eos_strings:
                    responses, unfinished_sequences = get_truncated_responses(responses, eos_strings)

                beam_scores = scorer(ScorerInput(response=responses, eos=(unfinished_sequences == 0)))

                _, beam_idx = torch.topk(beam_scores, w, dim=0, largest=True, sorted=True)
                beam_idx = beam_idx.repeat(k) # repeat beam_idx by n. successors

                # if unfinished_sequences.min().item() == 0: breakpoint()

                # reorder states
                input_ids = input_ids[beam_idx, :]
                unfinished_sequences = unfinished_sequences[beam_idx]

                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

                this_peer_finished = unfinished_sequences.max() == 0


        if return_dict_in_generate:
            return {
                "output_ids": input_ids[:w],
                "scores": beam_scores[:w],
            }
        else:
            return input_ids[0, None]
