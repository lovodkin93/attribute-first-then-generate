from src.inference.run_generative_clustering_model import run_generative_clustering_model
from src.train.load_models import load_fic_model
from src.inference.utils import _prepare_input, config_to_params_str
from src.inference.recovery import CalculateSimilarityByRouge, CalculateSimilarityBySimilarityScore, FiCBacktracker

import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
import logging
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq



def split_highlights_to_clusters(highlights):
    df = pd.DataFrame(highlights)
    
    # Create clusters
    statement_unique_id_column = 'scuSentCharIdx' if 'scuSentCharIdx' in df else 'scuSentence'
    clusters = df.groupby(statement_unique_id_column).apply(lambda rows: rows.to_dict('records')).to_list()
    
    # Order clusters based on the summary sentences order
    clusters = sorted(clusters, key=lambda cluster: cluster[0]['scuSentCharIdx'])
    
    return clusters
    

def do_clustering(use_cache, data, all_highlights, config):
    all_generated_sentences = None
    if config.get('do_clustering', False):
        all_clusters = None
        if use_cache and not config.get('rewrite_clustering_cache', False):
            try:
                if config['clustering_approach'] in ['backtrace_generation', 'generative_clustering']:
                    from src.inference.run_fic_model import load_from_cache
                    all_clusters, all_generated_sentences = load_from_cache(config)
                    
                elif config['clustering_approach'] == 'generative_highlight_n_cluster':
                    from src.inference.run_generative_highlight_n_cluster_model import load_from_cache
                    all_clusters, _ = load_from_cache(config, data)
                else:
                    raise NotImplementedError('clustering_approach cache not implemented')
            except:
                pass
            
        if all_clusters is None:
            if config['clustering_approach'] in ['backtrace_generation', 'generative_clustering']:
                dataset_type = 'fic' if config['clustering_approach'] == 'backtrace_generation' else config['clustering_approach']

                if dataset_type == 'fic':
                    generation_results, preprocessor = run_fic_model(data, all_highlights, config)
                elif dataset_type == 'generative_clustering':
                    generation_results, preprocessor = run_generative_clustering_model(data, all_highlights, config)
                else:
                    raise NotImplementedError('clustering_approach {clustering_approach} not implemented')

                if config['similarity_calculator'] == 'rouge':
                    similarity_calculator = CalculateSimilarityByRouge(highlight_sep_token=preprocessor.special_tokens_constants['highlights_separator'])
                elif config['similarity_calculator'] == 'semantic_similarity':
                    similarity_calculator = CalculateSimilarityBySimilarityScore(
                        model_name=config['similarity_calculator_model_name'],
                        threshold=config['similarity_calculator_threshold'],
                        matching_strategy=config['similarity_calculator_matching_strategy'],
                        matching_target=config['similarity_calculator_matching_target']
                    )
                else:
                    raise NotImplementedError('similarity_calculator {similarity_calculator} not implemented')
                
                fic_backtracker = FiCBacktracker(similarity_calculator, dataset_type=dataset_type, preprocessor=preprocessor)
                all_clusters = fic_backtracker.backtrace_fic_to_cluster(data, all_highlights, generation_results, config)
                all_clusters = [x[0] for x in all_clusters]
            elif config['clustering_approach'] == 'generative_highlight_n_cluster':
                from src.inference.run_generative_highlight_n_cluster_model import run_highlight_and_cluster_model
                all_clusters = run_highlight_and_cluster_model(data, config)
    else:
        logging.info('using gold clusters')
        all_clusters = [split_highlights_to_clusters(highlights) for highlights in all_highlights]

    if all_generated_sentences is not None and not config['do_summarization']:
        logging.info("updating scuSentence for all highlights")
        fic_backtracker = FiCBacktracker(None, None, None)
        assert len(data) == len(all_generated_sentences)
        for example, fic_generated_response in zip(data, all_generated_sentences):
            example['fic_response'] = fic_generated_response
            example['fic_response_sents'] = fic_backtracker.decompose_summary_to_sents(fic_generated_response) # TODO: save predicted_sents instead of recomputing

    return all_clusters




def get_results_path(config):
    return f"{config['highlights_representation_model_path']}/results/{config_to_params_str(config, '')}.json"


def run_fic_model(data, all_highlights, config):
    logging.info('running FiC model')

    model, tokenizer, device, preprocessor = load_fic_model(config)
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

            summaries = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
            assert len(summaries) == len(raw_examples), "num of summaries different between "
            for summary, example in zip(summaries, raw_examples):
                summarization_results.append({
                    "labels": example['response'],
                    "predicted": summary
                })

    if not config.get('filter_examples'):
        with open(get_results_path(config), "w") as f:
            json.dump(summarization_results, f, indent=4)

        
    return summarization_results, preprocessor



    
def load_from_cache(config):
    with open(get_results_path(config)) as f:
        cached_results = json.load(f)
        all_clusters = [x['clusters'] for x in cached_results]
        
        all_generated_sentences = None
        # Return generated sentences for the fic-generation experiment (in non-fic settings it's just a list of higlights separated by a special token)
        if config['clustering_approach'] == "backtrace_generation":
            all_generated_sentences = [x['summarization_result']['predicted'] for x in cached_results]
    
    return all_clusters, all_generated_sentences
