from transformers import LogitsProcessor
import torch
import numpy as np
import logging


class ConstrainedCopyLogitsProcessor(LogitsProcessor):
    """
    Constraints the highlights detection to adhere to input documents
    """
    
    def __init__(self, tokenizer, postprocessor, raw_examples, num_beams, min_highlights_generated):
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.raw_examples = raw_examples
        self.num_beams = num_beams
        self.min_highlights_generated = min_highlights_generated
        self.logger = logging.getLogger(__name__)
        
        
    def __call__(self, input_ids, scores):
        self.logger.debug("start __call__")
        input_ids = input_ids.detach().cpu()
        
        # Run over each beam
        for curr_beam_idx, curr_input_ids in enumerate(input_ids):
            example_idx = curr_beam_idx // self.num_beams
            
            curr_generated_highlight_str, curr_generated_highlight_tokens = self.find_current_generating_highlight(curr_input_ids)
            
            if curr_generated_highlight_str != '':
                allowed_next_generations = self.find_all_highlight_occurrences_in_text(curr_generated_highlight_str, example_idx)
                                
                allowed_next_generations_tokens = None
                if len(allowed_next_generations) > 0:
                    allowed_next_generations_tokens = self.find_next_allowed_generations_tokens(curr_generated_highlight_tokens, allowed_next_generations)

                self.update_scores(scores, curr_beam_idx, allowed_next_generations_tokens)

            num_highlights_generated = ((curr_input_ids == self.postprocessor.highlights_separator_idx) | (curr_input_ids == self.postprocessor.docs_sep_idx)).nonzero().shape[0]
            if num_highlights_generated < self.min_highlights_generated:
                scores[curr_beam_idx, self.tokenizer.eos_token_id] = -float('inf')

        self.logger.debug("end __call__")
        
        return scores

        
    def remove_special_tokens_from_list_of_ids(self, ids):
        return [x for x in ids if x.item() not in self.tokenizer.added_tokens_decoder]
    
    def find_current_generating_highlight(self, curr_input_ids):
        """
        At each iteration we are generating multiple highlights (e.g., "abc<highlight-sep>def"), we are currently interested in the one that is generating now
        """
        
        # self.logger.debug('start find_current_generating_highlight')
        generated_highlights = self.postprocessor.decode_pred(curr_input_ids)
        return generated_highlights[-1]

    def find_all_highlight_occurrences_in_text(self, curr_generated_highlight_str, example_idx):
        """
        Find all occurrences of the curr generating highlight in the text, take a few tokens ahead to see what is it allowed to generate
        """
        
        # self.logger.debug('start find_all_highlight_occurrences_in_text')
        allowed_next_generation = []
        for doc in self.raw_examples[example_idx]['documents']:
            # Find all occurrences of the highlight in the doc
            import re
            matches = re.finditer(re.escape(curr_generated_highlight_str), doc['rawDocumentText'])
            for match in matches:
                start_offset = match.start()
                allowed_next_generation.append(doc['rawDocumentText'][start_offset:start_offset+len(curr_generated_highlight_str)+30])
        
        return allowed_next_generation

    def find_next_allowed_generations_tokens(self, curr_generated_highlight_tokens, allowed_next_generations):
        """
        Find the index of the allowed next token to generate by removing the already generated ones (the input ids)
        """
        
        # self.logger.debug('start find_next_allowed_generations_tokens')
        curr_generated_highlight_tokens_without_special = self.remove_special_tokens_from_list_of_ids(curr_generated_highlight_tokens)
        allowed_next_generations_tokens = self.tokenizer(allowed_next_generations)['input_ids']
        allowed_next_generations_tokens_without_special = [self.remove_special_tokens_from_list_of_ids(np.array(x)) for x in allowed_next_generations_tokens]
        allowed_next_generations_token = []
        for allowed_next_generation_tokens in allowed_next_generations_tokens_without_special:
            generation_matches = True
            
            # Run over all the input ids to get the actual id
            allowed_next_generations_token_idx = 0
            for input_token_id in curr_generated_highlight_tokens_without_special:
                # Filter generations which do not start with the same tokens as input (can happen since we used regex to find the allowed_next_generation)
                # This is necessary mainly so we can find what's the next token idx, we can't add a token which doesn't share the prefix of the input
                if allowed_next_generation_tokens[allowed_next_generations_token_idx] != input_token_id:
                    generation_matches = False
                    break
                    
                allowed_next_generations_token_idx += 1

            did_generate_eos = allowed_next_generations_token_idx == len(allowed_next_generation_tokens)

            if generation_matches and not did_generate_eos:
                decoded = self.tokenizer.decode(allowed_next_generation_tokens[allowed_next_generations_token_idx])
                if any(decoded in allowed_next_generation for allowed_next_generation in allowed_next_generations):
                    allowed_next_generations_token.append(allowed_next_generation_tokens[allowed_next_generations_token_idx])
                
        return allowed_next_generations_token

    def update_scores(self, scores, curr_beam_idx, allowed_next_generations_token):
        # self.logger.debug('start update_scores')
        
        tokens_to_zero = torch.full((len(self.tokenizer.vocab),), -float('inf'), device=scores.device)
        
        if allowed_next_generations_token is not None:
            tokens_to_zero[allowed_next_generations_token] = scores[curr_beam_idx, allowed_next_generations_token]
        
        # Don't change values for stop copying current highlight / stop generating
        for special_token_idx in self.tokenizer.added_tokens_decoder.keys():
            if special_token_idx != self.tokenizer.unk_token_id and special_token_idx != self.tokenizer.pad_token_id and special_token_idx != self.tokenizer.mask_token_id:
                tokens_to_zero[special_token_idx] = scores[curr_beam_idx, special_token_idx]
            
        scores[curr_beam_idx] = tokens_to_zero
