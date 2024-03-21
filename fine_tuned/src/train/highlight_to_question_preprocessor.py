from collections import defaultdict
import logging
from typing import List, Optional, Tuple
import pandas as pd
import json
import re


# Used to separate between highlght spans when concatenating non-consecutive spans
HIGHLIGHT_SEP = "<HIGHLIGHT_SEP>"


class HighlightToQuestionPreprocessor:
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


    def preprocess_input(self, document, rows) -> str:
        """
        Converts input to str
        """

        title = self._put_highlights_in_input(document[f'source_title'], rows[rows['source_location'] == 'title'])
        text = self._put_highlights_in_input(document[f'source_text'], rows[rows['source_location'] == 'text'])
        
        input_text = f"{title}: {text}"

        return input_text

    def _put_highlights_in_input(self, curr_text, rows):
        """
        Converts input to str
        """

        if rows.empty:
            return curr_text
        
        rows = rows.drop_duplicates()
        rows = combine_intersecting_highlights(rows, curr_text)

        rows = rows.sort_values(by='end', ascending=False)

        # Add highlights to input
        for i, row in rows.iterrows():
            start_idx = row['start']
            end_idx = row['end']
            text = row['text']

            assert curr_text[start_idx:end_idx] == text, f"input_text[start_idx:end_idx] ({curr_text[start_idx:end_idx]}) != text ({text})"

            curr_text = curr_text[:start_idx] + f"{self.special_tokens_constants['highlight_start']}{text}{self.special_tokens_constants['highlight_end']}" + curr_text[end_idx:]

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
            query = examples['query'][i]
            documents = examples['documents'][i]
            set_of_highlights_in_context = examples['set_of_highlights_in_context'][i]

            if len(set_of_highlights_in_context) > 0:
                def source_id_to_inputs_outputs(rows):
                    source_id = rows.name
                    document = documents[source_id - 1]
                    input = self.preprocess_input(document, rows)
                    output = self.preprocess_output(query)
                    return input, output

                curr_inputs, curr_targets = zip(*pd.DataFrame(set_of_highlights_in_context).groupby('source_id').apply(source_id_to_inputs_outputs).tolist())

                inputs.extend(curr_inputs)
                targets.extend(curr_targets)

        model_inputs = self.tokenizer(
            inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)

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


def combine_intersecting_highlights(rows, orig_text: Optional[str], combine_same_sent_highlights: bool, group_by_summary_sentence: bool):
    """
    For example, if one highlight starts on index 0 and ends on index 22, and another starts on index 17 and ends on index 30, combine them to one row
    orig_text can be used to valdiate indices make sense
    """

    if 'documentFile' not in rows:
        rows['documentFile'] = rows.name
        
    assert rows['documentFile'].nunique() == 1

    any_row = rows.iloc[0]
    rows['docSpanOffsetStart'] = rows['docSpanOffsets'].apply(lambda x: x[0])
    rows['docSpanOffsetEnd'] = rows['docSpanOffsets'].apply(lambda x: x[1])
    # we need to sort by start because we later: (1) compare start offset to end offset and (2) only update the end of the last offset
    rows = rows.sort_values(by=['docSpanOffsetStart', 'docSpanOffsetEnd'], ascending=True)

 
    combined_rows = []
    curr_doc_span_offsets = []
    curr_doc_span_text = None
    curr_doc_sent_text = None
    curr_doc_sent_idx = None
    curr_doc_sent_char_idx = None
    curr_backtrace_scores = []
    curr_scu_sent_char_indices = []
    curr_scu_sentences = []
    curr_prefixes = []
    for row in rows.to_dict('records'):
        offset = [row['docSpanOffsetStart'], row['docSpanOffsetEnd']]
        doc_span_text = row['docSpanText']
        doc_sent_text = row['docSentText']
        doc_sent_idx = row.get('sent_idx')
        doc_sent_char_idx = row['docSentCharIdx']
        backtrace_score = row.get('FiCBacktraceMaxScore')
        scu_sent_char_index = row.get('scuSentCharIdx')
        scu_sentence = row.get('scuSentence')
        prefix = row.get('prefix')

        if len(curr_doc_span_offsets) == 0:
            curr_doc_span_offsets.append(offset)
            curr_doc_span_text = doc_span_text
            curr_doc_sent_text = doc_sent_text
            curr_doc_sent_idx = doc_sent_idx
            curr_doc_sent_char_idx = doc_sent_char_idx
            curr_backtrace_scores.append(backtrace_score)
            curr_scu_sent_char_indices.append(scu_sent_char_index)
            curr_scu_sentences.append(scu_sentence)
            curr_prefixes.append(prefix)
        else:
            are_same_sentence = doc_sent_char_idx == curr_doc_sent_char_idx
            intersecting_offset_idx = None
            for curr_offset_idx, curr_offset in enumerate(curr_doc_span_offsets):
                # TODO: verify adding +1, this is because of space separating two words verify noticable with the token classification model
                if offset[0] <= curr_offset[1] + 1:
                    intersecting_offset_idx = curr_offset_idx
                    break

            if intersecting_offset_idx is not None and are_same_sentence:
                if intersecting_offset_idx < len(curr_doc_span_offsets) - 1:
                    raise ValueError(f"{offset} intersects with {curr_doc_span_offsets[intersecting_offset_idx]} but is not the last offset in curr_offsets. Not supported currently, see if this error is thrown")

                # Update the last offset to the current offset end
                curr_doc_span_offsets[-1][1] = offset[1]

                # create new text based on new offset. We need to remove the docSentCharIdx to be able to query the current sentence and not the orig text
                doc_sent_char_idx = int(row['docSentCharIdx']) if isinstance(row['docSentCharIdx'], float) else row['docSentCharIdx']
                offset_to_use = [x - doc_sent_char_idx for x in curr_doc_span_offsets[-1]]
                new_text = curr_doc_sent_text[offset_to_use[0]:offset_to_use[1]]

                # Take all up to now besides last offset (because we updated it)
                curr_doc_span_text = HIGHLIGHT_SEP.join(curr_doc_span_text.split(HIGHLIGHT_SEP)[:-1])
                
                # Add the current text
                is_there_any_previous_text = len(curr_doc_span_offsets) > 1
                if is_there_any_previous_text:
                    curr_doc_span_text += HIGHLIGHT_SEP + new_text
                else:
                    curr_doc_span_text = new_text
                
                if orig_text is not None:
                    curr_text_based_on_orig_text = orig_text[offset[0]:offset[1]]
                    if len(curr_doc_span_text) != len(curr_text_based_on_orig_text) and doc_span_text.startswith('.'):
                        logging.warning(f'bug in dataset, {doc_span_text} which comes from docSpanText is not in correct length as docSpanOffsets. should be safe to ignore')
                    else:
                        assert curr_doc_span_text == curr_text_based_on_orig_text
            elif combine_same_sent_highlights and are_same_sentence:
                curr_doc_span_offsets.append(offset)
                curr_backtrace_scores.append(backtrace_score)
                curr_doc_span_text += (HIGHLIGHT_SEP + doc_span_text)
                curr_scu_sent_char_indices.append(scu_sent_char_index)
                curr_scu_sentences.append(scu_sentence)
                curr_prefixes.append(prefix)
            else:
                combined_row = {
                    'documentFile': any_row['documentFile'],
                    'docSentCharIdx': curr_doc_sent_char_idx,
                    'sent_idx': curr_doc_sent_idx,
                    'docSentText': curr_doc_sent_text,
                    'docSpanText':  curr_doc_span_text,
                    'docSpanOffsets': curr_doc_span_offsets,
                    'FiCBacktraceMaxScore': curr_backtrace_scores,
                }
                
                # If we don't group by summary sentence then after merging highlights these have no meaning
                if group_by_summary_sentence:
                    combined_row = {
                        **combined_row,
                        'scuSentCharIdx': any_row['scuSentCharIdx'],
                        'scuSentence': any_row['scuSentence'],
                        'prefix': any_row['prefix'],
                    }
                    
                    if 'scuSentIdx' in any_row:
                        combined_row['scuSentIdx'] = any_row['scuSentIdx']
                else:
                    combined_row = {
                        **combined_row,
                        'scuSentCharIdx': curr_scu_sent_char_indices,
                        'scuSentence': curr_scu_sentences,
                        'prefix': curr_prefixes
                    }
                    if 'scuSentIdx' in any_row:
                        combined_row['scuSentIdx'] = curr_scu_sent_char_indices
                    
                combined_rows.append(combined_row)

                curr_doc_span_offsets = [offset]
                curr_backtrace_scores = [backtrace_score]
                curr_scu_sent_char_indices = [scu_sent_char_index]
                curr_scu_sentences = [scu_sentence]
                curr_prefixes = [prefix]
                curr_doc_span_text = doc_span_text
                curr_doc_sent_text = doc_sent_text
                curr_doc_sent_idx = doc_sent_idx
                curr_doc_sent_char_idx = doc_sent_char_idx

    if len(curr_doc_span_offsets) > 0:
        combined_row = {
            'documentFile': any_row['documentFile'],
            'docSentCharIdx': curr_doc_sent_char_idx,
            'sent_idx': curr_doc_sent_idx,
            'docSentText': curr_doc_sent_text,
            'docSpanText':  curr_doc_span_text,
            'docSpanOffsets': curr_doc_span_offsets,
            'FiCBacktraceMaxScore': curr_backtrace_scores,
        }
        
        # If we don't group by summary sentence then after merging highlights these have no meaning
        if group_by_summary_sentence:
            combined_row = {
                **combined_row,
                'scuSentCharIdx': any_row['scuSentCharIdx'],
                'scuSentence': any_row['scuSentence'],
                'prefix': any_row['prefix'],
            }
            
            if 'scuSentIdx' in any_row:
                combined_row['scuSentIdx'] = any_row['scuSentIdx']
        else:
            combined_row = {
                **combined_row,
                'scuSentCharIdx': curr_scu_sent_char_indices,
                'scuSentence': curr_scu_sentences,
                'prefix': curr_prefixes
            }
            if 'scuSentIdx' in any_row:
                combined_row['scuSentIdx'] = curr_scu_sent_char_indices

        combined_rows.append(combined_row)

    for row in combined_rows:
        any_row = rows.iloc[0]
        row['documentFile'] = any_row['documentFile']

    return pd.DataFrame(combined_rows)
