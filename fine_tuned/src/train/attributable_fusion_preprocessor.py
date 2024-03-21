from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import logging
import json
import re
import numpy as np
import torch
from tqdm import tqdm

from src.train.highlight_to_question_preprocessor import HIGHLIGHT_SEP, combine_intersecting_highlights
from src.train.highlight_to_question_summ_preprocessor import parse_duc_source, parse_summ_dataset_format
from src.train import generation_e2e_preprocessor


class AttributableFusionPreprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self,
                 special_tokens_constants,
                 tokenizer,
                 ignore_pad_token_for_loss,
                 max_source_length,
                 max_target_length,
                 padding,
                 context_window: int,
                 only_before: bool,
                 device,
                 group_by_summary_sentence,
                 dataset_type: str
                 ):
        self.special_tokens_constants = special_tokens_constants
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.context_window = context_window
        self.only_before = only_before
        self.device = device
        self.group_by_summary_sentence = group_by_summary_sentence
        self.dataset_type = dataset_type

        # used for creating global attention mask
        self.special_tokens_ids = {}
        for key in special_tokens_constants:
            self.special_tokens_ids[key] = self.tokenizer.convert_tokens_to_ids(self.special_tokens_constants[key]) 



    # def doc_string_alternations(document_text, curr_text):
    #     document_text_changes = {"_":"-",
    #                             "`":"\'"}
    #     for old_str,new_str in document_text_changes.items():
    #         idxs = find_substring_indices(document_text.replace(old_str, new_str), curr_text)
    #         if idxs:
    #             return idxs, curr_text

    # def find_substring_indices(s, sub):
    #     indices = []
    #     i = s.find(sub)
    #     while i >= 0:
    #         indices.append(i)
    #         i = s.find(sub, i + 1)
    #     return indices
    #         return sentence.replace('_', '-').lower()




    def preprocess_cluster(self, rows, inputs, targets, documents, should_extract_targets, response, query):
        try:
            any_row = rows.iloc[0]
            
            curr_input = self.preprocess_input(rows, documents, query)

            inputs.append(curr_input)

            if should_extract_targets:
                if self.group_by_summary_sentence:
                    output = any_row['scuSentence']
                else:
                    if self.dataset_type == 'fic':
                        output = response
                    elif self.dataset_type in ['generative_clustering', 'generative_highlight_n_cluster']:
                        output = self.create_clustering_output(rows)
                    else:
                        raise ValueError(f"Unknown dataset type {self.dataset_type}")

                targets.append({
                    "text": output,
                    "alignments": rows
                })
        except Exception as e:
            logging.exception(f"Failed, skipping cluster")
            # raise ValueError("Failed") from e
            
    def preprocess_input(self, rows, documents, query) -> str:
        if self.dataset_type != 'generative_highlight_n_cluster':
            all_documents_sentences = rows.groupby('documentFile').apply(lambda highlights_rows_in_doc: self.preprocess_highlights(highlights_rows_in_doc, documents)).tolist()
            curr_input = self.special_tokens_constants['documents_separator'].join(all_documents_sentences)
        elif self.dataset_type == 'generative_highlight_n_cluster':
            docsep_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens_constants['documents_separator']) 
            curr_input, _, _ = generation_e2e_preprocessor.process_documents(documents, max_source_length=self.max_source_length, tokenizer=self.tokenizer, special_tokens_constants=self.special_tokens_constants, docsep_token_id=docsep_token_id)
        else:
            raise ValueError(f"Unknown dataset type {self.dataset_type}")
            

        if self.dataset_type == 'attributable_fusion':
            assert self.group_by_summary_sentence
            
            # Add prefix to input
            assert rows['prefix'].nunique() == 1
            any_row = rows.iloc[0]
            prefix = any_row['prefix']
            if prefix != "":
                curr_input = self.special_tokens_constants['highlights_prefix_separator'].join([curr_input, prefix])
            
        if query is None:
            return curr_input
        
        return f"Q: {query}" + self.special_tokens_constants['query_documents_separator'] + curr_input


    def preprocess_highlights(self, highlights_rows_in_doc, documents) -> str:        
        return add_highlights_to_source(highlights_rows_in_doc, documents, self.context_window, self.only_before, self.special_tokens_constants, group_by_summary_sentence=self.group_by_summary_sentence)


    def preprocess_examples(self, examples, should_extract_targets: bool):
        inputs, targets = [], []
        any_key = list(examples.keys())[0]
        for i in tqdm(range(len(examples[any_key]))):
            query = examples['query'][i] if 'query' in examples else None
            set_of_highlights_in_context = examples['set_of_highlights_in_context'][i]
            documents = examples['documents'][i]

            if len(set_of_highlights_in_context) > 0:
                curr_example_df = pd.DataFrame(set_of_highlights_in_context)
                if self.group_by_summary_sentence:
                    statement_unique_id_column = 'scuSentIdx' if 'scuSentIdx' in curr_example_df else ('scuSentCharIdx' if 'scuSentCharIdx' in curr_example_df else 'scuSentence')
                    curr_example_df.groupby(statement_unique_id_column).apply(lambda rows: self.preprocess_cluster(rows, inputs, targets, documents, should_extract_targets, response=None, query=query))
                else:
                    response = examples['response'][i] if should_extract_targets else None
                    self.preprocess_cluster(curr_example_df, inputs, targets, documents, should_extract_targets, response=response, query=query)

        return inputs, targets

    def preprocess_function(self, examples, is_training: bool, should_extract_targets: bool):
        convert_to_list = is_training
        
        inputs, targets = self.preprocess_examples(examples, should_extract_targets)
        if len(inputs) == 0:
            raise ValueError(f"Failed preprocess for unique_ids {examples['unique_id']}")

        if convert_to_list:
            model_inputs = self.tokenizer(
                inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)
        else:
            model_inputs = self.tokenizer(
                inputs, max_length=self.max_source_length, padding=self.padding, truncation=True, return_tensors='pt').to(self.device)

        # put global attention on seperator tokens
        global_attention_mask = []
        for curr_input_ids in model_inputs['input_ids']:
            curr_global_attention_mask = torch.zeros(len(curr_input_ids)).to(self.device)
            curr_global_attention_mask[0] = 1
            for key, token_id in self.special_tokens_ids.items():
                if isinstance(curr_input_ids, list):
                    curr_input_ids = torch.tensor(curr_input_ids)
                curr_global_attention_mask[curr_input_ids == token_id] = 1.0
            
            if convert_to_list:
                global_attention_mask.append(curr_global_attention_mask.tolist())
            else:
                global_attention_mask.append(curr_global_attention_mask)
        
        if convert_to_list:
            model_inputs['global_attention_mask'] = global_attention_mask
        else:
            model_inputs['global_attention_mask'] = torch.stack(global_attention_mask)
        
        if convert_to_list:
            model_inputs['input_ids'] = [x.tolist() if not isinstance(x, list) else x for x in model_inputs['input_ids']]

        if should_extract_targets:
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                target_texts = [target['text'] for target in targets]
                labels = self.tokenizer(
                    target_texts, max_length=self.max_target_length, padding="max_length", truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    

    def create_clustering_output(self, rows):
        statement_unique_id_column = 'scuSentIdx' if 'scuSentIdx' in rows else ('scuSentCharIdx' if 'scuSentCharIdx' in rows else 'scuSentence')
        all_summary_sents_highlights = rows.groupby(statement_unique_id_column).apply(lambda highlights_rows_in_doc: self.extract_summary_highlights_concatenated(highlights_rows_in_doc)).tolist()
        curr_output = self.special_tokens_constants['summary_sent_highlights_separator'].join(all_summary_sents_highlights)
        
        return curr_output
    
    def extract_summary_highlights_concatenated(self, highlights_rows):
        all_documents_highlights = highlights_rows.groupby('documentFile').apply(lambda highlights_rows_in_doc: self.extract_highlights_concatenated(highlights_rows_in_doc)).tolist()
        curr_output = self.special_tokens_constants['documents_separator'].join(all_documents_highlights)
        
        return curr_output

    
    def extract_highlights_concatenated(self, highlights_rows_in_doc):
        # Sort based on start offset
        assert highlights_rows_in_doc['documentFile'].nunique() == 1
        highlights_rows_in_doc['offset_start'] = highlights_rows_in_doc['docSpanOffsets'].apply(lambda x: x[0])
        highlights_rows_in_doc = highlights_rows_in_doc.sort_values('offset_start')
        
        # Dedup
        highlights_rows_in_doc = highlights_rows_in_doc[~highlights_rows_in_doc['docSpanOffsets'].duplicated()]
        
        return self.special_tokens_constants['highlights_separator'].join(highlights_rows_in_doc['docSpanText'])

        

def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """
    
    # raise ValueError("retrain this as I changed the doc-sep")

    special_tokens_constants = {}
    # T5 model has 100 special tokens by default
    if is_t5_model:
        special_tokens_constants['highlight_start'] = "<extra_id_1>"
        special_tokens_constants['highlight_end'] = "<extra_id_2>"
        special_tokens_constants['highlights_separator'] = "<extra_id_3>"
        special_tokens_constants['documents_separator'] = "<extra_id_4>"
        special_tokens_constants['highlights_prefix_separator'] = "<extra_id_5>"
        special_tokens_constants['summary_sent_highlights_separator'] = "<extra_id_6>"
        special_tokens_constants['query_documents_separator'] = "<extra_id_7>"
    else:
        special_tokens_constants['highlight_start'] = "<highlight_start>"
        special_tokens_constants['highlight_end'] = "<highlight_end>"
        special_tokens_constants['highlights_separator'] = "<highlights_separator>"
        special_tokens_constants['documents_separator'] = "<doc-sep>"
        # special_tokens_constants['documents_separator'] = "<documents_separator>"
        special_tokens_constants['highlights_prefix_separator'] = "<highlights_prefix_separator>"
        special_tokens_constants['summary_sent_highlights_separator'] = "<summary_sent_highlights_separator>"
        special_tokens_constants['query_documents_separator'] = "<query_documents_separator>"

    return special_tokens_constants


def get_closest_upper_and_lower_numbers(numbers, number):
    """
    Because of .explode based on .split('<SENT_SEP>'), we might have that the sentCharIdx is not updated, so take also the next sentence.
    Also, some sentences were originally split but now are not, so also look before / after.
    """
    return set([
        min([x for x in numbers if x >= number], default=number),
        max([x for x in numbers if x <= number], default=number)
    ])


def get_sent(document, highlight):
    sentence = None
    sent_idx = None
    for possible_doc_char_idx in get_closest_upper_and_lower_numbers(document['docSentCharIdxToSentIdx'], highlight['docSentCharIdx']):
        possible_sent_idx = document['docSentCharIdxToSentIdx'].index(possible_doc_char_idx)
        possible_sentence = document['documentText'][possible_sent_idx]
        if all(parse_duc_source(highlight_part) in parse_duc_source(possible_sentence) for highlight_part in highlight['docSpanText'].split(HIGHLIGHT_SEP)):
            sentence = possible_sentence
            sent_idx = possible_sent_idx
            break
    if sentence is None:
        raise ValueError(f"Could not find sentence for highlight '{highlight['docSpanText']}', {highlight['docSentCharIdx']}, {possible_sentence}")
    return sentence, sent_idx



def add_highlights_to_source(highlights_rows_in_doc, documents, context_window, only_before, special_tokens_constants, group_by_summary_sentence: bool) -> str:
    """
    Adds highlights to source documents
    Highlights_rows_in_doc is a dataframe of all the highlights in a single document.
    The idea is to avoid repeating lines of the same document when taking a window.
    """

    any_row = highlights_rows_in_doc.iloc[0]
    
    assert highlights_rows_in_doc['documentFile'].nunique() == 1
    document_file = any_row['documentFile']
    
    document = [x for x in documents if x['documentFile'] == document_file]
    assert len(document) == 1, f"Other than one document found for documentFile {document_file}"
    document = document[0]
    
    all_inputs = []

    # Do this here for LFQA (instead of create_datasets.ipynb) because otherwise when doing this when creating the dataset it fails to load it
    # Also, for inference time (predicted highlights which were combined)
    highlights_rows_in_doc = parse_summ_dataset_format(highlights_rows_in_doc, span_separator=HIGHLIGHT_SEP)
    highlights_rows_in_doc = combine_intersecting_highlights(highlights_rows_in_doc, orig_text=None, combine_same_sent_highlights=True, group_by_summary_sentence=group_by_summary_sentence)

    all_sentences_with_markers = {}
    for _, highlight_row in highlights_rows_in_doc.iterrows():
        sent_idx = highlight_row['sent_idx']
        doc_sent_text = highlight_row['docSentText']

        sent_with_h_markers = add_highlight_markers_to_sent(highlight_row, doc_sent_text, special_tokens_constants)

        # Shouldn't happen because we combine same sent highlights. The second term is for the case of the same sentence being repeated, due to a bug in the dataset
        assert sent_idx not in all_sentences_with_markers or all_sentences_with_markers[sent_idx] == sent_with_h_markers

        all_sentences_with_markers[sent_idx] = sent_with_h_markers

    
    sentences = get_context_sentences_by_window(context_window, only_before, document, all_sentences_with_markers)

    sentences_str = '\n'.join(sentences)

    all_inputs.append(sentences_str)

    return special_tokens_constants['highlights_separator'].join(all_inputs)

def add_highlight_markers_to_sent(highlight_row, doc_sent_text, special_tokens_constants):
    """
    replace sentence text by highlight the OIE in the sentence
    """
    
    docSentCharIdx = highlight_row['docSentCharIdx']
    
    new_offsets_without_doc_sent_char_idx = []
    for offset in highlight_row['docSpanOffsets']:
        new_offset = [x-docSentCharIdx for x in offset]
        new_offset = [int(x) for x in new_offset]
        new_offsets_without_doc_sent_char_idx.append(new_offset)
    offsets = np.array(new_offsets_without_doc_sent_char_idx).reshape((-1,2))

    for offset in offsets[::-1]:
        doc_sent_text = doc_sent_text[:offset[0]] + special_tokens_constants['highlight_start'] + doc_sent_text[offset[0]:offset[1]] + special_tokens_constants['highlight_end'] + doc_sent_text[offset[1]:]
    return doc_sent_text


def get_context_sentences_by_window(window: int, only_before: bool, document: list, all_sentences_with_markers):
    """
    We want to take the context of all sentences around the sentence with the highlight, but if there is a sentence with highlight we want to take it.
    """

    window_after = 0 if only_before else window

    # 1. first find out which sentences to we want to include
    all_sentences_to_include = []
    for sent_idx in all_sentences_with_markers.keys():
        sent_idx = int(sent_idx) if isinstance(sent_idx, float) else sent_idx
        # For each index in the window make sure there exists a sentence
        sentences_to_include = [i for i in range(max(0, sent_idx-window), sent_idx+window_after+1) if len(document['documentText']) > i]
        all_sentences_to_include.extend(sentences_to_include)

    # 2. then run over all sentences in document and add them only if they are in the include list. if they have a version with highlights, then use it
    sentences = []
    for sent_idx, doc_sent_text in enumerate(document['documentText']):
        if sent_idx in all_sentences_with_markers:
            sentences.append(all_sentences_with_markers[sent_idx])
        elif sent_idx in all_sentences_to_include:
            sentences.append(doc_sent_text)

    return sentences
