from Few_shot_experiments.utils import remove_spaces_and_punctuation
from src.inference.constrained_copy_logits_processor import ConstrainedCopyLogitsProcessor
from src.inference.run_highlights_detection import postprocess_predicted_highlights
from src.train.load_models import load_generative_clustering_model, load_generative_highlight_n_cluster_model
from src.train.highlight_to_question_preprocessor import HIGHLIGHT_SEP
from src.train.highlight_to_question_summ_preprocessor import combine_intersecting_highlights
from src.inference.utils import _prepare_input, config_to_params_str
from nltk.corpus import stopwords
from nltk import word_tokenize

import spacy
from tqdm import tqdm
import evaluate
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import logging
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from collections import defaultdict


def get_results_path(config):
    return f"{config['highlights_representation_model_path']}/results/{config_to_params_str(config, '')}.json"


def load_from_cache(config, data):
    with open(get_results_path(config)) as f:
        all_clusters = json.load(f)
        all_clusters = [x['clusters'] for x in all_clusters]
    
    # Temp here because it was not run in teh first time
    assert len(data) == len(all_clusters)
    new_all_clusters = []
    for clusters, example in zip(all_clusters, data):
        new_clusters = []
        for cluster in clusters:
            if len(cluster) > 0:
                # We want to group highlights here because we already created the clusters
                new_cluster = postprocess_predicted_highlights(cluster, example, group_by_summary_sentence=True)
                new_clusters.append(new_cluster)
                
        new_all_clusters.append(new_clusters)
    
    return new_all_clusters
    
    
    return all_clusters

def run_highlight_and_cluster_model(data, config):
    logging.info('running generative clustering model')
    
    model, tokenizer, device, preprocessor, postprocessor = load_generative_highlight_n_cluster_model(config)
    
    all_clusters = []
    summarization_results = []

    # Transformers dataset format
    dataset = Dataset.from_dict(pd.DataFrame(data).to_dict('list'))

    dataset = dataset.map(
        lambda row: preprocessor.preprocess_function(
            row,
            is_training=True,  # We can't batch without padding
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
            if config.get('min_highlights_generated') is not None:
                logits_processor = [ConstrainedCopyLogitsProcessor(
                        tokenizer=tokenizer,
                        postprocessor=postprocessor,
                        raw_examples=raw_examples,
                        num_beams=num_beams,
                        min_highlights_generated=config['min_highlights_generated']
                    )]

            batch_preds = model.generate(
                max_new_tokens=preprocessor.max_target_length,
                num_beams=num_beams,
                logits_processor=logits_processor,
                **inputs
            )
            
            summaries = tokenizer.batch_decode(batch_preds)
            
            # clean special tokens but not the separators
            clean_summaries = []
            for summary in summaries:
                clean_summary = summary.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').replace(tokenizer.bos_token, '')
                clean_summaries.append(clean_summary)
            summaries = clean_summaries
            
            # Find spans for generated highlights
            for summary_sent_idx, pair in enumerate(zip(summaries, raw_examples)):
                summary, example = pair
                
                clusters = []
                summary_sents = summary.split(preprocessor.special_tokens_constants['summary_sent_highlights_separator'])
                for summary_sent in summary_sents:
                    flattened_predictions = [x for y in summary_sent.split(preprocessor.special_tokens_constants['documents_separator']) for x in y.split(preprocessor.special_tokens_constants['highlights_separator'])]
                    cluster = postprocessor.find_offsets_to_list_of_spans(flattened_predictions, example['documents'])
                    for highlight_obj in cluster:
                        highlight_obj['scuSentIdx'] = summary_sent_idx
                        
                    if len(cluster) > 0:
                        cluster = postprocess_predicted_highlights(cluster, example, group_by_summary_sentence=True)
                        
                    clusters.append(cluster)
                all_clusters.append(clusters)
            
    assert len(all_clusters) == len(data)
    for clusters, example in zip(all_clusters, data):
        summarization_results.append({
            "labels": example['response'],
            "clusters": clusters
        })
            
    if not config.get('filter_examples'):
        with open(get_results_path(config), "w") as f:
            json.dump(summarization_results, f, indent=4)


    return all_clusters


