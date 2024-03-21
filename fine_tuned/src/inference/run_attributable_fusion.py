import torch
import evaluate
import nltk
import json
import numpy as np

from src.inference.no_repeat_ngram_from_prefix_logits_processor import NoRepeatNGramFromPrefixLogitsProcessor
from src.inference.utils import config_to_params_str
from src.train.load_models import load_attributable_fusion_model


def do_summarization(use_cache, data, all_clusters, config):
    if config.get('do_summarization', False):
        # If previous components were not cached we should also avoid in this phase using cached
        if use_cache and not config.get('rewrite_clustering_cache', False):
            try:
                generation_results = load_generation_results_from_cache(get_results_path(config))
                # can remove this 
                save_autoais_format(get_results_path(config, prefix="autoais_"), data, [x['summarization_results'] for x in generation_results], all_clusters)

                return generation_results
            except:
                pass
            
        generation_results = run_attributable_fusion_model(data, all_clusters, config)
    # Use the results from the FiC process as the summary
    else:
        assert config['clustering_approach'] == "backtrace_generation", "do_summarization=False only supported for backtrace_generation, otherwise the scuSentence is the gold scuSentence and not generated ones in the FiC process"
        generation_results = []
        for example in data:
            generation_results.append({
                "query": example.get('query'),
                "labels": example['response'],
                "predicted": example['fic_response'],
                "predicted_sents": example['fic_response_sents']
            })
        
        if not config.get('filter_examples'):
            with open(get_results_path(config), "w") as f:
                json.dump([{'summarization_results': generation_result, 'set_of_highlights_in_context': clusters} for generation_result, clusters in list(zip(generation_results, all_clusters))], f, indent=4)
            
            save_autoais_format(get_results_path(config, prefix="autoais_"), data, generation_results, all_clusters, config)
            
    

def get_results_path(config, prefix=""):
    return f"{config['attributable_fusion_model_path']}/results/{config_to_params_str(config, prefix)}.json"


def run_attributable_fusion_model(data, all_clusters, config):
    summarization_results = []
    
    attributable_fusion_model, attributable_fusion_tokenizer, device, attributable_fusion_preprocessor = load_attributable_fusion_model(config)
    
    assert len(data) == len(all_clusters)
    for example, clusters in zip(data, all_clusters):
        # Since we generate sentence-by-sentence, we need to keep track of the summary (also to use it as the prefix for the next cluster)
        summary = ""
        summary_sents = []
        for cluster in clusters:
            for highlight in cluster:
                highlight['prefix'] = summary
                
            summary_sentence = None
            if len(cluster) > 0:
                
                cluster_input = attributable_fusion_preprocessor.preprocess_function(
                    {
                        "unique_id": [example['unique_id']],
                        "documents": [example['documents']],
                        "set_of_highlights_in_context": [cluster],
                        "query": [example.get('query')]
                    },
                    is_training=False,
                    should_extract_targets=False
                )
                with torch.no_grad():
                    prefix_tokens_ids = attributable_fusion_preprocessor.tokenizer(summary)['input_ids'][:-1]
                    
                    summary_sentence = attributable_fusion_model.generate(
                        max_new_tokens=attributable_fusion_preprocessor.max_target_length,
                        num_beams=config['num_beams'],
                        logits_processor=[NoRepeatNGramFromPrefixLogitsProcessor(
                            attributable_fusion_model.generation_config.no_repeat_ngram_size,
                            prefix_tokens_ids,
                            attributable_fusion_preprocessor.tokenizer,
                        )],
                        **cluster_input
                    )
                    summary_sentence = attributable_fusion_tokenizer.decode(summary_sentence[0], skip_special_tokens=True)
                    summary += summary_sentence + " "
            summary_sents.append(summary_sentence)
                
        summarization_results.append({
            "query": example.get('query'),
            "labels": example['response'],
            "predicted": summary,
            "predicted_sents": summary_sents
        })

    write_to_cache(config, data, summarization_results, all_clusters)
        
    return summarization_results

def write_to_cache(config, data, summarization_results, all_clusters):
    if not config.get('filter_examples'):
        with open(get_results_path(config), "w") as f:
            json.dump([{'summarization_results': summarization_result, 'set_of_highlights_in_context': clusters} for summarization_result, clusters in list(zip(summarization_results, all_clusters))], f, indent=4)
    
        save_autoais_format(get_results_path(config, prefix="autoais_"), data, summarization_results, all_clusters, config)


def load_generation_results_from_cache(results_path):
    with open(results_path) as f:
        generation_results = json.loads(f.read())
    
    return generation_results

def evaluate_generation(results_path):
    generation_results = load_generation_results_from_cache(results_path)
        
    metric = evaluate.load("rouge", seed=42)
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    decoded_preds = [x['summarization_results']['predicted'] for x in generation_results]
    decoded_labels = [x['summarization_results']['labels'] for x in generation_results]
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)    
    
    metric = evaluate.load("bertscore")
    bertscore_results = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en", verbose=True) #, rescale_with_baseline=True)
    result['bertscore'] = np.mean(bertscore_results['f1'])
    
    metric = evaluate.load("meteor")
    meteor_results = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result['meteor'] = meteor_results['meteor']
    
    result = {k: round(v * 100, 4) for k, v in result.items()}

    result['len'] = len(decoded_preds)
    result['results_path'] = results_path

    print(result)
    return result

def align_clusters_with_summarization_results(clusters, summarization_result, config):
    """
    When in fic_generation mode, it is possible that some sentences do not have a cluster.
    Since we didn't save empty clusters, the assertion `len(summarization_result['predicted_sents']) == len(clusters)` can fail.
    In this method we re-add the empty clusters
    """
    
    if config['do_summarization']:
        return clusters
    
    scu_sentence_to_cluster = {cluster[0]['scuSentence']: cluster for cluster in clusters}
    
    
    new_clusters = []
    for predicted_sent in summarization_result['predicted_sents']:
        if predicted_sent in scu_sentence_to_cluster:
            new_clusters.append(scu_sentence_to_cluster[predicted_sent])
        else:
            new_clusters.append([])
    
    return new_clusters

def save_autoais_format(results_path, data, summarization_results, all_clusters, config):    
    # For e2e usecase
    if all_clusters is None:
        all_clusters = [None] * len(summarization_results)
              
    predicted_examples = []

    # save format for auto ais (same format as our data)
    assert len(data) == len(summarization_results)
    assert len(data) == len(all_clusters)
    for example, summarization_result, clusters in zip(data, summarization_results, all_clusters):
        # Match highlights format
        if clusters is not None:
            clusters = align_clusters_with_summarization_results(clusters, summarization_result, config)
            assert len(summarization_result['predicted_sents']) == len(clusters)
            for summary_sent_idx, pair in enumerate(zip(summarization_result['predicted_sents'], clusters)):
                summary_sentence, cluster = pair
                
                if cluster is not None:
                    for highlight in cluster:
                        # Override the summary sentence generated in the FiC setting with the one generated in the attributable fusion setting
                        highlight['scuSentence'] = summary_sentence
                        # The original index might be higher than summary_sent_idx because some summary sentences from the FiC were not used
                        highlight['scuSentIdx'] = summary_sent_idx
                        # Remove unused columns
                        highlight.pop('scuSentCharIdx')
        
        predicted_example = {
            "unique_id": example['unique_id'],
            "documents": example['documents'],
            "set_of_highlights_in_context": [highlight for cluster in clusters for highlight in cluster] if clusters is not None else None,
            "response": summarization_result['predicted'],
            "query": example.get('query')
        }
                
        predicted_examples.append(predicted_example)

    with open(results_path, "w") as f:
        f.writelines([f"{json.dumps(example)}\n" for example in predicted_examples])

