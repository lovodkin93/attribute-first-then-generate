from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import json
import re
from tqdm import tqdm
import torch
from src.train.attributable_fusion_preprocessor import add_highlights_to_source

from src.train.generation_e2e_preprocessor import get_document_truncated_text_with_offset_mapping, process_documents
from src.train.highlight_to_question_preprocessor import HIGHLIGHT_SEP


class HighlightDetectionPreprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self,
                 special_tokens_constants,
                 tokenizer,
                 data_args,
                 max_target_length,
                 padding,
                 device,
                 add_global_attention
                 ):
        self.special_tokens_constants = special_tokens_constants
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_source_length = self.data_args['max_source_length'] if isinstance(self.data_args, dict) else self.data_args.max_source_length
        self.device = device
        self.max_target_length = max_target_length
        self.padding = padding
        self.add_global_attention = add_global_attention
        
        # used for creating global attention mask
        self.special_tokens_ids = {}
        for key in special_tokens_constants:
            self.special_tokens_ids[key] = self.tokenizer.convert_tokens_to_ids(self.special_tokens_constants[key]) 

    def preprocess_output(self, rows, documents) -> str:
        """
        Converts output to str
        """

        is_output_original_text_with_highlights = False
        all_documents_strs = []
        rows = pd.DataFrame(rows)
        # Iterate over documents to keep a consistent order
        for document in documents:
            highlights_rows_in_doc = rows[rows['documentFile'] == document['documentFile']]
            
            # Remove dups
            highlights_rows_in_doc = highlights_rows_in_doc[~highlights_rows_in_doc[['docSpanOffsets']].duplicated()]
            
            if is_output_original_text_with_highlights:
                document_strs = add_highlights_to_source(highlights_rows_in_doc, documents, context_window=100, only_before=False, special_tokens_constants=self.special_tokens_constants)
            # Only highlights
            else:
                highlights_rows_in_doc = highlights_rows_in_doc.sort_values('docSentCharIdx')
                
                flattened_highlights = [x for y in highlights_rows_in_doc['docSpanText'].tolist() for x in y.split(HIGHLIGHT_SEP)]
                document_strs = self.special_tokens_constants['highlights_separator'].join(flattened_highlights)
                
            all_documents_strs.append(document_strs)
            # all_documents_strs = rows.groupby('documentFile').apply(lambda highlights_rows_in_doc: add_highlights_to_source(highlights_rows_in_doc, documents, context_window=100, only_before=False, special_tokens_constants=self.special_tokens_constants)).tolist()
            
        curr_output = self.special_tokens_constants['documents_separator'].join(all_documents_strs)
        return curr_output
    
    def preprocess_output_for_classification(self, rows, documents):
        rows = pd.DataFrame(rows)
        all_documents_labels = rows.groupby('documentFile').apply(lambda highlights_rows_in_doc: self.document_to_classifaction_labels(highlights_rows_in_doc, documents)).tolist()
                    
        return all_documents_labels
    
    def document_to_classifaction_labels(self, highlights_rows_in_doc, documents):
        """
        Converts output to list of binary decisions per word
        """

        any_row = highlights_rows_in_doc.iloc[0]
        
        assert highlights_rows_in_doc['documentFile'].nunique() == 1
        document_file = any_row['documentFile']
        
        document = [x for x in documents if x['documentFile'] == document_file]
        assert len(document) == 1, f"More than one document found for documentFile {document_file}"
        document = document[0]
        
        labels = []
        # char_counter = 0
        document_text, _, _, offset_mapping = get_document_truncated_text_with_offset_mapping(document, documents, self.data_args.max_source_length, self.tokenizer)
        for token_offset in offset_mapping:                        
            is_token_highlighted = False
            for _, highlight_row in highlights_rows_in_doc.iterrows():
                for offset in highlight_row['docSpanOffsets']:
                    are_not_intersecting = offset[0] >= token_offset[1] or token_offset[0] >= offset[1]
                    if not are_not_intersecting:
                        is_token_highlighted = True
                        break
                    
                if is_token_highlighted:
                    break
            
            labels.append({
                "word": document_text[token_offset[0]:token_offset[1]],
                "start_char": token_offset[0],
                "end_char": token_offset[1],
                "is_word_highlighted": is_token_highlighted
            })
            
            
        return labels

    def preprocess_function(self, examples, is_training: bool, is_classification_task: bool, should_extract_targets: bool):
        convert_to_list = is_training
        query_exists = 'query' in examples and examples['query'][0] is not None

        inputs, targets = self.preprocess_examples(examples, is_classification_task, should_extract_targets)

        input_ids = []
        attention_mask = []
        for input_obj in inputs:
            input_ids.append(input_obj['input_ids'])
            attention_mask.append(input_obj['attention_mask'])
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if not convert_to_list:
            model_inputs['input_ids'] = torch.tensor(model_inputs['input_ids']).to(self.device)
            model_inputs['attention_mask'] = torch.tensor(model_inputs['attention_mask']).to(self.device)
        #     model_inputs = self.tokenizer(
        #         inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)
        # else:
        #     model_inputs = self.tokenizer(
        #         inputs, max_length=self.max_source_length, padding=self.padding, truncation=True, return_tensors='pt').to(self.device)


        if self.add_global_attention:
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
            
        if should_extract_targets:
            if not is_classification_task:
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
            else:
                all_labels = self.convert_classification_objs_to_labels(model_inputs, targets, query_exists)

                model_inputs["labels"] = all_labels

        return model_inputs
    
    def convert_classification_objs_to_labels(self, model_inputs, targets, query_exists: bool):
        """
        For easier analysis we keep objects after preprocessing the targets instead of just False or True.
        This method converts these objects to a list of lists of labels, and also validates the lengths.
        """
        
        all_labels = []
        for example_i, pair in enumerate(zip(model_inputs['input_ids'], targets)):
            input_ids, target = pair
            
            # Each example starts with <bos>
            labels = [-100]
            words = ["<s>"]
            
            # Update the labels based on the query
            num_tokens_that_q_removed_from_last_document = 0
            if query_exists:
                # This actually counts the first <s> but doesn't count the separating <doc-sep> between the q and the documents, so no need to remove / add 1
                q_end_idx = input_ids.index(self.special_tokens_ids['documents_separator'])
                
                # prepend -100 tokens to the labels equal to the query so we have correct loss calculation
                for _ in range(q_end_idx):
                    labels.append(-100)
                    words.append("q")
            
            for document_labels in target:
                for word_label in document_labels:
                    words.append(word_label['word'])
                    labels.append(int(word_label['is_word_highlighted']))
                    
                # for doc separator (and the last iteartion of the loop will be for the eos)
                words.append("<doc-sep>")
                labels.append(0)
                
            # also have to remove tokens from the last document since the query took space
            if query_exists:
                num_spare_tokens_before_adding_q = self.max_source_length - len(labels) + q_end_idx
                num_tokens_that_q_removed_from_last_document = q_end_idx - num_spare_tokens_before_adding_q
                
                if num_tokens_that_q_removed_from_last_document > 0:
                    labels = labels[:-num_tokens_that_q_removed_from_last_document]

            # For debugging, run `list(zip(self.tokenizer.convert_ids_to_tokens(input_ids), words))`
            assert len(input_ids) == len(labels), f"Input and target lengths don't match for example {example_i}"
            
            all_labels.append(labels)
            
        return all_labels
            
    
    def preprocess_examples(self, examples, is_classification_task: bool, should_extract_targets: bool):
        inputs, targets = [], []

        any_key = list(examples.keys())[0]
        for i in tqdm(range(len(examples[any_key]))):
            query = examples['query'][i] if 'query' in examples else None
            documents = examples['documents'][i]
            
            inputs.append(self.preprocess_input(query, documents))
            if should_extract_targets:
                set_of_highlights_in_context = examples['set_of_highlights_in_context'][i]
                
                if not is_classification_task:
                    targets.append(self.preprocess_output(set_of_highlights_in_context, documents))
                else:
                    targets.append(self.preprocess_output_for_classification(set_of_highlights_in_context, documents))
                

        return inputs, targets

    def preprocess_input(self, query, documents):
        """
        Converts input to str
        """

        doc_sep_token_id = self.special_tokens_ids['documents_separator']
        _, input_ids, attention_mask = process_documents(documents, max_source_length=self.max_source_length, tokenizer=self.tokenizer, special_tokens_constants=self.special_tokens_constants, docsep_token_id=doc_sep_token_id)
        
        def build_input(tokenizer, input_ids, attention_mask):
            input = []
            new_attention_mask = []
            if tokenizer.bos_token_id is not None:
                input.append(tokenizer.bos_token_id)
                new_attention_mask.append(1)
            
            input.extend(input_ids)
            new_attention_mask.extend(attention_mask)
            
            input.append(tokenizer.eos_token_id)
            new_attention_mask.append(1)
            
            return {
                "input_ids": input,
                "attention_mask": new_attention_mask
            }
        

        # Return text with prefix
        if query is None:
            return {**build_input(self.tokenizer, input_ids, attention_mask)}
        
        query_inputs = self.tokenizer.encode(f"Q: {query}")[1:-1]
        
        # To see why this is painfully necessary, see documentation in `get_document_truncated_text_with_offset_mapping`
        input_ids = query_inputs + [doc_sep_token_id] + input_ids
        attention_mask = (len(query_inputs) * [1]) + [1] + attention_mask
        
        input_ids = input_ids[:self.max_source_length - 2]
        attention_mask = attention_mask[:self.max_source_length - 2]
        assert len(input_ids) == len(attention_mask)
            
        return {**build_input(self.tokenizer, input_ids, attention_mask)}


def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """

    special_tokens_constants = {}
    # T5 model has 100 special tokens by default
    if is_t5_model:
        special_tokens_constants['highlight_start'] = "<extra_id_1>"
        special_tokens_constants['highlight_end'] = "<extra_id_2>"
        special_tokens_constants['highlights_separator'] = "<extra_id_3>"
        special_tokens_constants['documents_separator'] = "<extra_id_4>"
    else:
        special_tokens_constants['highlight_start'] = "<highlight_start>"
        special_tokens_constants['highlight_end'] = "<highlight_end>"
        special_tokens_constants['highlights_separator'] = "<highlights_separator>"
        special_tokens_constants['documents_separator'] = "<doc-sep>"

    return special_tokens_constants
