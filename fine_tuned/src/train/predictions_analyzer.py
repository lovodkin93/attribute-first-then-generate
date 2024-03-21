import os
import sys
from datasets import load_metric
import pandas as pd
import numpy as np    
import json
import re


class PredictionsAnalyzer:
    """
    Extracts an analyzed result for each prediction instead of an aggregate of all predictions
    """

    def __init__(self, tokenizer, preprocessor, output_dir: str, is_classification_task: bool) -> None:
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.output_dir = output_dir
        self.is_classification_task = is_classification_task

    def write_predictions_to_file(self, predictions, dataset, df, data_file_path: str):
        objects = []
        for input_example, prediction in zip(dataset, predictions):
            if not self.is_classification_task:
                objects.append({
                    "input": self.tokenizer.decode(input_example['input_ids']),
                    "label": self.tokenizer.decode(input_example['labels']) if 'labels' in input_example else None,
                    "pred": self.tokenizer.decode(self.remove_pad_tokens(prediction))
                })
            else:
                objects.append({
                    "input": self.tokenizer.decode(input_example['input_ids']),
                    "label": input_example['labels'] if 'labels' in input_example else None,
                    "pred": self.remove_pad_tokens(prediction)
                })

        self._save_to_file(objects, data_file_path)

    def remove_pad_tokens(self, prediction_tokens):
        """
        We want to calculate the num of tokens without the padding
        """

        return [token for token in prediction_tokens if token not in [self.tokenizer.pad_token_id, -100]]

    def _save_to_file(self, objects, data_file_path):
        df = pd.DataFrame(objects)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        output_prediction_file = os.path.join(
            self.output_dir, f"generated_predictions_{'.'.join(data_file_path.replace('/', '_').split('.')[:-1])}.csv")
        df.to_csv(output_prediction_file, index=False)

