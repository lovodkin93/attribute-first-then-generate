import logging
import torch
import evaluate
import nltk
import json
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import logging
import os
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from collections import defaultdict
from src.inference.run_attributable_fusion import save_autoais_format

from src.inference.utils import _prepare_input
from src.train.load_models import load_e2e_model


def get_results_path(config, prefix=''):
    model_path = config['e2e_model_path']
    if not os.path.exists(model_path):
        model_path = config['data_path']
    file_name_unique_id = '.'.join(config['data_file_path'].replace('/','_').split('.')[:-1])
    return f"{model_path}/results/{prefix}results_{file_name_unique_id}.json"


def run_e2e_model(data, config):
    summarization_results = []
    
    model, tokenizer, device, preprocessor = load_e2e_model(config)
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

            # Necessary for labels for analysis
            raw_examples = data[step*batch_size:(step+1)*batch_size]

            batch_preds = model.generate(
                max_new_tokens=preprocessor.max_target_length,
                num_beams=config['num_beams'],
                **inputs
            )
            
            summaries = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
            for summary, example in zip(summaries, raw_examples):
                summarization_results.append({
                    "labels": example['response'],
                    "predicted": summary
                })
        
    if not config.get('filter_examples'):
        with open(get_results_path(config), "w") as f:
            json.dump([{'summarization_results': summarization_result} for summarization_result in summarization_results], f, indent=4)
            
        save_autoais_format(get_results_path(config, prefix="autoais_"), data, summarization_results, [])
        
    return summarization_results

