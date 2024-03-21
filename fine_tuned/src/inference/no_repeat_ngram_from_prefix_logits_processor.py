from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
import torch
import numpy as np
import logging


# Based on NoRepeatNGramLogitsProcessor from transformers
class NoRepeatNGramFromPrefixLogitsProcessor(LogitsProcessor):
    """
    When generating sentence-by-sentence, this mimics the no repeat ngram used in the end-to-end generation (see NoRepeatNGramLogitsProcessor).
    Note that this means no repeating n-grams only from the prefix, but repeating other parts in the input is allowed
    """
    
    def __init__(self, ngram_size: int, prefix_tokens_ids, tokenizer):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size
        self.prefix_tokens_ids = prefix_tokens_ids
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Remove the bos of the input ids (we are going to add one from the prefix token ids)
        input_ids = input_ids[:, 1:]
        prefix_tokens_ids_repeated = torch.tensor(self.prefix_tokens_ids).unsqueeze(0).repeat((len(input_ids), 1)).to(input_ids.device)
        input_ids = torch.hstack([prefix_tokens_ids_repeated, input_ids])
        
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores