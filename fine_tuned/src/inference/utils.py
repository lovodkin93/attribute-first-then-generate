from typing import Any, Mapping, Union, Optional
import torch
import logging
import os
import json
import datetime
import pandas as pd

from src.train.generation_e2e_preprocessor import get_document_truncated_text_with_offset_mapping
from src.train.load_models import load_highlights_detection_model

# Modified from Trainer._prepare_input . used to move the inputs to GPU.
def _prepare_input(data: Union[torch.Tensor, Any], device) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        # NEW - commented out deepspeed check
        # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
        #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's inputs are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
        return data.to(**kwargs)
    return data


def config_to_params_str(config, prefix=""):
    file_name_unique_id = '.'.join(config['data_file_path'].replace('/','_').split('.')[:-1])[-6:]
    min_highlights_generated = config.get('min_highlights_generated')
    highlights_detection_num_beams = config.get('highlights_detection_num_beams')
    sim_calc_model_name = config.get('similarity_calculator_model_name')
    if sim_calc_model_name is not None:
        sim_calc_model_name = sim_calc_model_name.replace('/', '_')[-3:]
    similarity_calculator = config.get('similarity_calculator')
    if similarity_calculator is not None:
        similarity_calculator = similarity_calculator[-3:]
    matching_strategy = config.get('similarity_calculator_matching_strategy')
    if matching_strategy is not None:
        matching_strategy = matching_strategy[-3:]
        
    clustering_approach = config.get('clustering_approach')
        
    return f"{prefix}results_{file_name_unique_id}__do_h_{config['do_highlights']}__min_h_{min_highlights_generated}__h_beams__{highlights_detection_num_beams}__do_c_{config['do_clustering']}__{clustering_approach}__sim_calc_{similarity_calculator}_{sim_calc_model_name}_{config.get('similarity_calculator_threshold')}_{matching_strategy}_{config.get('similarity_calculator_matching_target')}__do_s_{config.get('do_summarization')}"



def get_documents_truncation_indices(data, config):
    """
    To fairly compare between different models / oracle settings, we want to truncate the documents based on one model.
    This will return the index for each example for each document in which it is truncated
    """
    
    logging.info("calculating documents truncation indices")
    
    model, tokenizer, device, highlights_detection_preprocessor, postprocessor = load_highlights_detection_model(config, config['highlights_detection_model_path'])
    
    all_documents_truncation_indices = []
    for example in data:
        documents_ids_to_truncation_idx = {}
        # example_highlights = []
        # orig_highlights = example['set_of_highlights_in_context']
        documents = example['documents']
        for doc in documents:
            _, _, _, offset_mappings = get_document_truncated_text_with_offset_mapping(doc, documents, highlights_detection_preprocessor.max_source_length, tokenizer)
            last_idx_used = offset_mappings[-1][1]
            
            documents_ids_to_truncation_idx[doc['documentFile']] = last_idx_used
            
        all_documents_truncation_indices.append(documents_ids_to_truncation_idx)
        
    return all_documents_truncation_indices



def get_run_timestamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S%z")


def save_results(all_results_path, results, config, run_timestamp: Optional[str]):
    if run_timestamp is None:
        run_timestamp = get_run_timestamp()
    results['time'] = run_timestamp
    results['config'] = json.dumps(config)
    all_results = load_results_file(all_results_path)
    all_results.append(results)
    pd.DataFrame(all_results).to_csv(all_results_path, index=False)

    
def load_results_file(all_results_path):
    if os.path.exists(all_results_path):
        all_results_df = pd.read_csv(all_results_path)
        all_results = all_results_df.to_dict('records')
    else:
        all_results = []
        
    return all_results
