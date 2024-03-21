from src.inference.recovery import parse_cross_sent_highlight
from src.train.generation_e2e_preprocessor import get_document_truncated_text_with_offset_mapping
from src.train.highlight_to_question_preprocessor import HIGHLIGHT_SEP
from src.train.load_models import load_highlights_detection_model, load_highlights_detection_model_w_encoder
from src.train.highlight_to_question_summ_preprocessor import combine_intersecting_highlights, parse_summ_dataset_format
from Few_shot_experiments.run_content_selection_MDS import adapt_highlights_to_doc_alignments
from src.inference.constrained_copy_logits_processor import ConstrainedCopyLogitsProcessor
from src.inference.utils import _prepare_input, config_to_params_str

import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import logging
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq

logger = logging.getLogger(__name__)


def do_highlights(use_cache, data, all_documents_truncation_indices, config, orig_data):
    # highlights are not created with clusters
    if config.get('clustering_approach') == 'generative_highlight_n_cluster':
        return None
    
    if config.get('do_highlights', False):
        all_highlights = None
        if use_cache:
            try:
                all_highlights = load_highlights_detection_from_cache(config)
                
                # If we loaded highlights from cache, we need to also filter the highlights
                if config.get('filter_examples') is not None:
                    all_highlights = [highlight for x, highlight in zip(orig_data, all_highlights) if x['topic'] in config.get('filter_examples')]
            except:
                pass
        
        if all_highlights is None:
            all_highlights = run_highlights_detection_model(data, config)
    else:
        logging.info('using gold highlights')
        stats_about_removed_highlights = []
        all_highlights = []
        
        # Filter data based the amount of information the models would see
        assert len(data) == len(all_documents_truncation_indices)
        for example, documents_truncation_indices in zip(data, all_documents_truncation_indices):
            example_highlights = []
            orig_highlights = example['set_of_highlights_in_context']
            documents = example['documents']
            for doc in documents:
                last_idx_used = documents_truncation_indices[doc['documentFile']]
                                
                example_highlights.extend([highlight for highlight in orig_highlights if highlight['docSentCharIdx'] < last_idx_used and highlight['documentFile'] == doc['documentFile']])
                
            stats_about_removed_highlights.append({
                "num_of_orig_highlights": len(orig_highlights),
                "num_of_example_highlights": len(example_highlights),
                "num_of_removed_highlights": len(orig_highlights) - len(example_highlights),
                "unique_id": example['unique_id']
            })
                
            all_highlights.append(example_highlights)

        percent_removed_highlights = sum([x['num_of_removed_highlights'] for x in stats_about_removed_highlights]) / sum([x['num_of_orig_highlights'] for x in stats_about_removed_highlights])
        logging.info(f"{percent_removed_highlights} of highlights removed from gold")
        
        # Save gold highlights in same format for analysis
        from src.inference.run_highlights_detection import get_results_path as highlights_detection_get_results_path
        with open(highlights_detection_get_results_path(config), "w") as f:
            json.dump(all_highlights, f, indent=4)

    
    assert len(data) == len(all_highlights)
    for example, highlights in zip(data, all_highlights):
        example['set_of_highlights_in_context'] = highlights
        

    return all_highlights    

    


def get_results_path(config):
    return f"{config['highlights_detection_model_path']}/results/{config_to_params_str(config, '')}.json"

def run_highlights_detection_model(data, config):
    if config['highlights_approach'] == 'w_encoder':
        return run_highlights_detection_w_encoder_model(data, config)
    else:
        return run_highlights_detection_model(data, config)

   

def run_highlights_detection_model(data, config):
    logger.debug("starting run_highlights_detection_model")
    model, tokenizer, device, highlights_detection_preprocessor, postprocessor = load_highlights_detection_model(config, config['highlights_detection_model_path'])
    all_preds = []
    
    # Transformers dataset format
    dataset = Dataset.from_dict(pd.DataFrame(data).to_dict('list'))

    dataset = dataset.map(
        lambda row: highlights_detection_preprocessor.preprocess_function(
            row,
            is_training=True,  # We can't batch without padding
            is_classification_task=False,
            should_extract_targets=False
        ),
        batched=True,
        remove_columns=dataset.features.keys()  # Remove raw dataset columns
    )
    batch_size = 10
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,  # Not sure if necessary, but also defaults to true in TrainingArguments.
        pin_memory_device=str(device)
    )
    
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            logging.info(f"Starting batch {step+1}/{len(dataloader)}")
            inputs = _prepare_input(inputs, device)
            
            # Necessary for constrained decoding
            raw_examples = data[step*batch_size:(step+1)*batch_size]
            
            num_beams = config['highlights_detection_num_beams']
            
            logits_processor = []
            min_highlights_generated = config['min_highlights_generated']
            if min_highlights_generated is not None:
                logits_processor = [ConstrainedCopyLogitsProcessor(
                        tokenizer=tokenizer,
                        postprocessor=postprocessor,
                        raw_examples=raw_examples,
                        num_beams=num_beams,
                        min_highlights_generated=config['min_highlights_generated']
                    )]
            
            batch_preds = model.generate(
                max_new_tokens=highlights_detection_preprocessor.max_target_length,
                num_beams=num_beams,
                logits_processor=logits_processor,
                **inputs
            )

            all_preds_set_of_highlights_in_context, _ = postprocessor.extract_prediction(batch_preds.detach().cpu(), None, raw_examples=raw_examples)
                
            assert len(all_preds_set_of_highlights_in_context) == len(raw_examples)
            for set_of_highlights_in_context, example in zip(all_preds_set_of_highlights_in_context, raw_examples): 
                if len(set_of_highlights_in_context) > 0:
                    # We don't want to group highlights here because we didn't create the clusters yet
                    set_of_highlights_in_context = postprocess_predicted_highlights(set_of_highlights_in_context, example, group_by_summary_sentence=False)
                    
                    all_preds.append(set_of_highlights_in_context)
                else:
                    all_preds.append([])

    if not config.get('filter_examples'):
        with open(get_results_path(config), "w") as f:
            json.dump(all_preds, f, indent=4)
    
    return all_preds

def postprocess_predicted_highlights(set_of_highlights_in_context, example, group_by_summary_sentence: bool):
    """
    Predicted highlights require special care. We need to:
    - Explode highlights
    - Split across sents highlights
    - Combine intersecting or same row
    """
    
    # Explode highlight. validate_orig_text is not necessary here because there are highlights we inserted
    set_of_highlights_in_context = parse_summ_dataset_format(pd.DataFrame(set_of_highlights_in_context), span_separator=HIGHLIGHT_SEP, validate_orig_text=False).to_dict('records')
    
    # Split across sents highlights
    new_set_of_highlights_in_context = []
    for highlight in set_of_highlights_in_context:
        new_set_of_highlights_in_context.extend(parse_cross_sent_highlight(highlight, example))
    set_of_highlights_in_context = new_set_of_highlights_in_context

    # Combine intersecting or same row
    highlights_df = pd.DataFrame(set_of_highlights_in_context)
    highlights_df['docSentCharIdx'] = highlights_df['docSentCharIdx'].apply(int)
    if 'prefix' not in highlights_df:
        highlights_df['prefix'] = ''
    set_of_highlights_in_context = pd.DataFrame(highlights_df).groupby('documentFile').apply(lambda rows: combine_intersecting_highlights(rows, orig_text=None, combine_same_sent_highlights=True, group_by_summary_sentence=group_by_summary_sentence)).to_dict('records')
        
    return set_of_highlights_in_context

def load_highlights_detection_from_cache(config):
    with open(get_results_path(config)) as f:
        all_preds = json.load(f)
    
    return all_preds
    
    
def run_highlights_detection_w_encoder_model(data, config):
    model, tokenizer, device, highlights_detection_preprocessor = load_highlights_detection_model_w_encoder(config, config['highlights_detection_model_path'])
    all_preds = []
    for example in data:
        model_inputs = highlights_detection_preprocessor.preprocess_function(
            {
                "topic": [example['topic']],
                "documents": [example['documents']]
            },
            is_training=False,
            is_classification_task=True,
            should_extract_targets=False
        )
        
        with torch.no_grad():
            results = model(**model_inputs)
            preds = results['logits'][0].to('cpu')
            preds = np.argmax(preds, axis=1)
            flattened_preds = torch.tensor(preds).view(-1)
            
            set_of_highlights_in_context = preds_to_set_of_highlights_in_context(example['documents'], highlights_detection_preprocessor, flattened_preds, model_inputs['input_ids'][0])
            
            # Combine intersecting or same row
            set_of_highlights_in_context = pd.DataFrame(set_of_highlights_in_context).groupby('documentFile').apply(lambda rows: combine_intersecting_highlights(rows, orig_text=None, combine_same_sent_highlights=True, group_by_summary_sentence=False)).to_dict('records')
            
            all_preds.append(set_of_highlights_in_context)

    
    return all_preds


def preds_to_set_of_highlights_in_context(documents, preprocessor, flattened_preds, input_ids):
    set_of_highlights_in_context = []
    
    # Remove bos and eos
    flattened_preds = flattened_preds[1:-1]
    input_ids = input_ids[1:-1]
    
    doc_sep_token_id = preprocessor.special_tokens_ids['documents_separator']
    doc_sep_indices = torch.where(input_ids == doc_sep_token_id)[0].tolist()
    
    for doc_idx, doc in enumerate(documents):
        doc_start_offset = 0
        doc_end_offset = len(flattened_preds)
        if len(doc_sep_indices) > 0:
            if doc_idx > 0:
                doc_start_offset = doc_sep_indices[doc_idx-1]+1
            if doc_idx < len(doc_sep_indices):
                doc_end_offset = doc_sep_indices[doc_idx]
                
        curr_flattened_preds = flattened_preds[doc_start_offset:doc_end_offset]
        curr_input_ids = input_ids[doc_start_offset:doc_end_offset]

        new_doc_text, _, _, offset_mapping = get_document_truncated_text_with_offset_mapping(doc, documents, preprocessor.data_args, preprocessor.tokenizer)
        
        assert len(curr_input_ids) == len(curr_flattened_preds)
        assert len(curr_input_ids) == len(offset_mapping)
        for curr_input, pred, offset in zip(curr_input_ids, curr_flattened_preds, offset_mapping):
            if pred == 1:
                # Find lower closest
                doc_sent_char_idx = max([x for x in doc['docSentCharIdxToSentIdx'] if x <= offset[0]], default=offset[0])
                doc_sent_char_idx_to_doc_sent_idx = {doc_sent_char_idx: doc_sent_idx for doc_sent_idx, doc_sent_char_idx in enumerate (doc['docSentCharIdxToSentIdx'])}
                sent_idx = doc_sent_char_idx_to_doc_sent_idx[doc_sent_char_idx]
                
                set_of_highlights_in_context.append({
                    'documentFile': doc['documentFile'],
                    'docSpanOffsets': offset,
                    'docSpanText': new_doc_text[offset[0]:offset[1]],
                    'sent_idx': sent_idx,
                    'docSentText': doc['documentText'][sent_idx],
                    'docSentCharIdx': doc_sent_char_idx
                })
        
            
    return set_of_highlights_in_context
