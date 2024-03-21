from collections import defaultdict
import logging
from typing import List, Tuple
import pandas as pd
import json
import re

from src.train.highlight_to_question_preprocessor import combine_intersecting_highlights
from Few_shot_experiments.utils import remove_spaces_and_punctuation


class HighlightToQuestionSummPreprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self,
                 special_tokens_constants,
                 tokenizer,
                 data_args,
                 max_target_length,
                 padding,
                 ):
        self.special_tokens_constants = special_tokens_constants
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_target_length = max_target_length
        self.padding = padding


    def preprocess_input(self, document_text, rows, document) -> str:
        """
        Converts input to str
        """

        return self._put_highlights_in_input(document_text, rows, document)

    def _put_highlights_in_input(self, curr_text, rows, document):
        """
        Converts input to str
        """

        if rows.empty:
            return curr_text
        
        # rows = rows.drop_duplicates()

        rows = parse_summ_dataset_format(rows)
        rows = combine_intersecting_highlights(rows, curr_text)

        rows = rows.sort_values(by='end', ascending=False)

        # Add highlights to input
        for i, row in rows.iterrows():
            text = row['text']

            start_idx = row['start']
            end_idx = row['end']
            
            if len(curr_text[start_idx:end_idx]) == len(text):
                assert curr_text[start_idx:end_idx] == text, f"input_text[start_idx:end_idx] ({curr_text[start_idx:end_idx]}) != text ({text})"
            # Known problem where the dot is not aligned correctly to the docSpanText. The curr_text is the correct value, which is based on docSpanOffsets
            elif text.startswith('.') or curr_text[start_idx:end_idx].endswith('.'):
                logging.warning(f'bug in dataset, {text} which comes from docSpanText is not in correct length as docSpanOffsets')
            else:
                raise ValueError('unexpected length of text')

            curr_text = curr_text[:start_idx] + f"{self.special_tokens_constants['highlight_start']}{curr_text[start_idx:end_idx]}{self.special_tokens_constants['highlight_end']}" + curr_text[end_idx:]

        return curr_text


    def preprocess_output(self, query) -> str:
        """
        Converts output to str
        """

        return query


    def preprocess_function(self, examples):
        inputs, targets = [], []
        any_key = list(examples.keys())[0]
        for i in range(len(examples[any_key])):
            # No output in this dataset
            output = None
            documents = examples['documents'][i]
            set_of_highlights_in_context = examples['set_of_highlights_in_context'][i]

            if len(set_of_highlights_in_context) > 0:
                def source_id_to_inputs_outputs(rows):
                    document_file = rows.name
                    document = [d for d in documents if document_file == d['documentFile']][0]
                    input = self.preprocess_input(document['documentText'], rows, document)
                    return input, output

                df = pd.DataFrame(set_of_highlights_in_context)
                curr_inputs, curr_targets = zip(*df.groupby('documentFile').apply(source_id_to_inputs_outputs).tolist())

                inputs.extend(curr_inputs)
                targets.extend(curr_targets)

        model_inputs = self.tokenizer(
            inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)
        
        # It is possible that there is not output in some cases, such as inference
        should_parse_output = targets[0] is not None

        if should_parse_output:
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets, max_length=self.max_target_length, padding="max_length", truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]

        return model_inputs


def parse_duc_source(sentence):
    """
    duc highlights has mistakes, these fixes to the source should match it to the highlights
    """

    return remove_spaces_and_punctuation(sentence).lower()



def _parse_doc_span_text(text_to_split, orig_text, separator, validate_orig_text: bool) -> str:
    """
    This method does two things:
    1. Split by separator
    2. Validate that we didn't split things that were not supposed to be split (see below). This can be turn off with validate_orig_text = False, necessary when generating higlights.
    We split `text_to_split` based on the separator "..." or "<SPAN_SEP>".
    However, sometimes the text also has "...", so we need to check for each two parts if their combination was in the orignal text.
    If yes, we join them back together.
    """

    # text_parts_after_split = [y for x in text_to_split.split(separator) for y in x.split('...')]
    text_parts_after_split = text_to_split.split(separator)
    
    if not validate_orig_text:
        return text_parts_after_split

    text_parts = []
    skip_next_part = False
    for i in range(len(text_parts_after_split)):
        if skip_next_part:
            skip_next_part = False
            continue
        
        curr_text = text_parts_after_split[i]

        assert parse_duc_source(curr_text) in parse_duc_source(orig_text) or curr_text.startswith('.') , f"curr_text ({curr_text}) not in orig_text ({orig_text})"
        
        next_text_after_split = None
        if len(text_parts_after_split) > i + 1:
            next_text_after_split = text_parts_after_split[i + 1]

            assert next_text_after_split in orig_text or next_text_after_split.startswith('.'), f"next_text_after_split ({next_text_after_split}) not in orig_text ({orig_text})"

        if next_text_after_split is not None and curr_text + separator + next_text_after_split in orig_text:
            text_parts.append(curr_text + separator + next_text_after_split)
            skip_next_part = True
        else:
            text_parts.append(curr_text)

    return text_parts


    

    
def parse_summ_dataset_format(rows, span_separator: str = '...', validate_orig_text: bool = True):
    """
    Explodes rows with multiple highlights into separate rows
    """

    # Break the highlight row into multiple rows if it's not continuous
    rows['docSpanOffsets_parsed'] = rows['docSpanOffsets'].apply(lambda ranges_strs: ranges_strs.split(';') if isinstance(ranges_strs, str) else ranges_strs)
    # The idea of comparing to docSentText is to avoid cases where the separator of "..." was in the original text
    rows['docSpanText_parsed'] = rows.apply(lambda row: _parse_doc_span_text(row['docSpanText'], row['docSentText'], span_separator, validate_orig_text=validate_orig_text), axis=1)
    rows = rows.explode(['docSpanOffsets_parsed', 'docSpanText_parsed'])
    rows['docSpanText'] = rows['docSpanText_parsed']
    rows['docSpanOffsets'] = rows['docSpanOffsets_parsed'].apply(lambda range_str: [int(x) for x in range_str.split(',')] if isinstance(range_str, str) else range_str)

    rows['docSpanOffsetStart'] = rows['docSpanOffsets'].apply(lambda x: x[0])
    rows['docSpanOffsetEnd'] = rows['docSpanOffsets'].apply(lambda x: x[1])

    return rows


def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """

    special_tokens_constants = {}
    # T5 model has 100 special tokens by default
    if is_t5_model:
        special_tokens_constants['highlight_start'] = "<extra_id_1>"
        special_tokens_constants['highlight_end'] = "<extra_id_2>"
    else:
        special_tokens_constants['highlight_start'] = "<highlight_start>"
        special_tokens_constants['highlight_end'] = "<highlight_end>"

    return special_tokens_constants

def find_non_consecutive_substrings(main_string, sub_string):
    """
    In case of mismatch between a sub string and a main string, this function finds all the possible indices of the sub string in the main string.
    """

    memo = {}

    def find_indices_recursive(s, sub, start):
        if not sub:
            return [()]

        if (start, sub) in memo:
            return memo[(start, sub)]

        indices = []
        for i in range(start, len(s)):
            if s[i] == sub[0]:
                for rest in find_indices_recursive(s, sub[1:], i + 1):
                    indices.append((i,) + rest)
        
        memo[(start, sub)] = indices
        return indices

    index_combinations = find_indices_recursive(main_string, sub_string, 0)
    start_end_indices = [(comb[0], comb[-1]) for comb in index_combinations if comb]

    return list(set([main_string[s:e+1].strip() for (s,e) in start_end_indices]))
