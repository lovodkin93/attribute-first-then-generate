from tqdm import tqdm
import pandas as pd
import torch


class GenerationE2EPreprocessor:
    def __init__(self,
                 special_tokens_constants,
                 tokenizer,
                 ignore_pad_token_for_loss,
                 max_source_length,
                 max_target_length,
                 padding,
                 device
                 ):
        self.special_tokens_constants = special_tokens_constants
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.device = device
        
        # used for creating global attention mask
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens_constants['documents_separator']) 
    
    def preprocess_function(self, examples, is_training: bool, should_extract_targets: bool):
        convert_to_list = is_training

        inputs, targets = self.preprocess_examples(examples, should_extract_targets)

        if convert_to_list:
            model_inputs = self.tokenizer(
                inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)
        else:
            model_inputs = self.tokenizer(
                inputs, max_length=self.max_source_length, padding=self.padding, truncation=True, return_tensors='pt').to(self.device)

        # put global attention on doc seperator token
        global_attention_mask = []
        for curr_input_ids in model_inputs['input_ids']:
            curr_global_attention_mask = torch.zeros(len(curr_input_ids)).to(self.device)
            curr_global_attention_mask[0] = 1
            curr_global_attention_mask[torch.tensor(curr_input_ids) == self.docsep_token_id] = 1.0
            
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
                labels = self.tokenizer(
                    targets, max_length=self.max_target_length, padding="max_length", truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def preprocess_examples(self, examples, should_extract_targets: bool):
        inputs, targets = [], []
        any_key = list(examples.keys())[0]
        for i in tqdm(range(len(examples[any_key]))):
            query = examples['query'][i] if 'query' in examples else None
            curr_input = self.preprocess_input(examples['documents'][i], query)
            inputs.append(curr_input)
            targets.append(examples['response'][i] if should_extract_targets else None)
    
        return inputs, targets

    def preprocess_input(self, documents, query):
        doc_texts, _, _ = process_documents(documents, max_source_length=self.max_source_length, tokenizer=self.tokenizer, special_tokens_constants=self.special_tokens_constants, docsep_token_id=self.docsep_token_id)
        if query is None:
            return doc_texts
        
        return f"Q: {query}" + self.special_tokens_constants['documents_separator'] + doc_texts


def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """

    special_tokens_constants = {}
    special_tokens_constants['documents_separator'] = "<doc-sep>"

    return special_tokens_constants

def reconstruct_doc_text_from_document_text(doc):
    """
    rawDocumentText is not reliable for DUC, so we reconstruct it from documentText (specifically orig_documents['2006']['D0605']['documents']['AW19980603.1001'] starts with space)
    """

    doc_text = ""
    for sent_idx, pair in enumerate(zip(doc['docSentCharIdxToSentIdx'], doc['documentText'])):
        sentCharIdx, sent = pair

        while len(doc_text) < sentCharIdx:
            doc_text += " "
        
        doc_text += sent
    
    return doc_text

def get_document_truncated_text_with_offset_mapping(doc, documents, max_source_length, tokenizer):
    # doc_text = reconstruct_doc_text_from_document_text(doc)
    doc_text = doc['rawDocumentText']
    
    # Shorten doc text so all documents enter the input
    max_source_length = max_source_length
    doc_max_length = max_source_length // len(documents)
    doc_text_tokenized = tokenizer(
                        doc_text,
                        truncation=True,
                        max_length=doc_max_length,
                        return_offsets_mapping=True
                    )
    # -2 because the last token is eos_token </s>
    last_offset_mapping = doc_text_tokenized['offset_mapping'][-2]
    last_doc_text_idx = last_offset_mapping[1]
    
    new_doc_text = doc_text[:last_doc_text_idx]
    
    # Try running this code with max_length=4 and max_length=5:
    # ```[tokenizer.decode(x) for x in tokenizer(['a ’'], truncation=True, max_length=5)['input_ids']]```
    # Unexpectedely, the token ’ is only half-removed and still has an offset.
    # This makes the following assertion fail, which is why in cases that we care about offset mapping (highlights encoder detection)
    #    we also need to use these original ids and can't return only the text.
    # assert len(tokenizer.encode(new_doc_text, truncation=True)) == len(doc_text_tokenized['offset_mapping'])
        
    # remove bos and eos tokens
    input_ids = doc_text_tokenized['input_ids'][1:-1]
    attention_mask = doc_text_tokenized['attention_mask'][1:-1]
    offset_mapping = doc_text_tokenized['offset_mapping'][1:-1]
        
    return new_doc_text, input_ids, attention_mask, offset_mapping
        


def process_documents(documents, max_source_length, tokenizer, special_tokens_constants, docsep_token_id):
    doc_texts = []
    all_input_ids = []
    all_attention_mask = []
    for doc in documents:
        doc_text, input_ids, attention_mask, _ = get_document_truncated_text_with_offset_mapping(doc, documents, max_source_length, tokenizer)
        
        doc_texts.append(doc_text)
        all_input_ids.extend(input_ids)
        all_attention_mask.extend(attention_mask)
        
        all_input_ids.append(docsep_token_id)
        all_attention_mask.append(1)

    doc_texts = special_tokens_constants['documents_separator'].join(doc_texts)
    return doc_texts, all_input_ids[:-1], all_attention_mask[:-1]
