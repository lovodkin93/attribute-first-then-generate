import json
import logging
import sys

# One of the imports is messing with the logs (they stop showing in console) so needs to be here
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


from src.inference.run_highlights_detection import do_highlights
from src.inference.run_fic_model import do_clustering
from src.inference.run_attributable_fusion import do_summarization, evaluate_generation, get_results_path
from src.inference.utils import get_documents_truncation_indices, get_run_timestamp, save_results
from src.train.utils import read_minified_config


def load_data(config):
    data_file_path = config['data_file_path']

    with open(data_file_path) as f:
        data = [json.loads(line) for line in f.readlines()]

    return data


def main(config):
    logging.info(f'config {config}')
    
    orig_data = load_data(config)
    data = orig_data
        
    all_documents_truncation_indices = get_documents_truncation_indices(data, config)
    
    use_cache = config.get('use_cache', True)
    
    if config.get('filter_examples') is not None:
        data = [x for x in data if x['topic'] in config.get('filter_examples')]
    
    all_highlights = do_highlights(use_cache, data, all_documents_truncation_indices, config, orig_data)        
    all_clusters = do_clustering(use_cache, data, all_highlights, config)
    do_summarization(use_cache, data, all_clusters, config)
    return evaluate_generation(get_results_path(config))


def hp_training(config):
    file_name_unique_id = '.'.join(config['data_file_path'].replace('/','_').split('.')[:-1])
    all_results_path = f"{config['attributable_fusion_model_path']}/all_results_{file_name_unique_id}.csv"
    
    run_timestamp = get_run_timestamp()

    for highlights_detection_num_beams in [2, 5, 10, 15]:
        for min_highlights_generated in [None, 0, 10, 20, 30]:
            for similarity_calculator_matching_strategy in ["highest_score"]: #, "highlight"]:
                try:
                    config['highlights_detection_num_beams'] = highlights_detection_num_beams
                    config['min_highlights_generated'] = min_highlights_generated
                    config['similarity_calculator_matching_strategy'] = similarity_calculator_matching_strategy
                    results = main(config)
                except Exception as e:
                    curr_path = get_results_path(config)
                    logging.exception(f'failed results_path {curr_path}')
                    
                    results = {
                        "results_path": curr_path,
                        "error": str(e)
                    }
                    
                save_results(all_results_path, results, config, run_timestamp)
                


if __name__ == '__main__':
    config_file_path = sys.argv[-1]

    config = read_minified_config(config_file_path)

    hp_tuning = config.get('hp_tuning', False)
    
    if not hp_tuning:
        results = main(config)
        file_name_unique_id = '.'.join(config['data_file_path'].replace('/','_').split('.')[:-1])
        all_results_path = f"{config['attributable_fusion_model_path']}/results/inference_results_{file_name_unique_id}.csv"
        save_results(all_results_path, results, config, None)
    else:
        hp_training(config)
