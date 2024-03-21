from Few_shot_experiments.utils import remove_spaces_and_punctuation
from src.train.load_models import load_generative_clustering_model
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



def run_generative_clustering_model(data, all_highlights, config):
    logging.info('running generative clustering model')
    
    model, tokenizer, device, preprocessor = load_generative_clustering_model(config)
    
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

            batch_preds = model.generate(
                max_new_tokens=preprocessor.max_target_length,
                num_beams=config['num_beams'],
                **inputs
            )
            
            summaries = tokenizer.batch_decode(batch_preds)
            
            # clean special tokens but not the separator
            clean_summaries = []
            for summary in summaries:
                clean_summary = summary.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '').replace(tokenizer.bos_token, '')
                clean_summaries.append(clean_summary)
            summaries = clean_summaries
            
            assert len(summaries) == len(raw_examples)
            for summary, example in zip(summaries, raw_examples):
                summarization_results.append({
                    "labels": example['response'],
                    "predicted": summary
                })
        
    if not config.get('filter_examples'):
        with open(get_results_path(config), "w") as f:
            json.dump(summarization_results, f, indent=4)


    return summarization_results, preprocessor


