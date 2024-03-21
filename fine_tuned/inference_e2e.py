import json
import logging
import pandas as pd
import sys
from src.inference.run_e2e_model import get_results_path, run_e2e_model

# One of the imports is messing with the logs (they stop showing in console) so needs to be here
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


from src.inference.run_attributable_fusion import evaluate_generation
from src.train.utils import read_minified_config

def load_data(config):
    data_file_path = config['data_file_path']

    with open(data_file_path) as f:
        data = [json.loads(line.replace("\n", "")) for line in f.readlines()]

    return data


def split_highlights_to_clusters(highlights):
    df = pd.DataFrame(highlights)
    
    # Create clusters
    statement_unique_id_column = 'scuSentCharIdx' if 'scuSentCharIdx' in df else 'scuSentence'
    clusters = df.groupby(statement_unique_id_column).apply(lambda rows: rows.to_dict('records')).to_list()
    
    # Order clusters based on the summary sentences order
    clusters = sorted(clusters, key=lambda cluster: cluster[0]['scuSentCharIdx'])
    
    return clusters
    

def main(config):
    # console_handler = logging.StreamHandler()
    # logging.getLogger().addHandler(console_handler)
    

    logging.info(f'config {config}')
    
    data = load_data(config)
    # data = [x for x in data if x['topic'] == 'test81']
    
    use_cache = False
    if not use_cache:
        run_e2e_model(data, config)

    evaluate_generation(get_results_path(config))
        

if __name__ == '__main__':
    config_file_path = sys.argv[-1]

    config = read_minified_config(config_file_path)

    main(config)

