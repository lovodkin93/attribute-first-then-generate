from nltk.corpus import stopwords
from nltk import word_tokenize
from src.train.highlight_to_question_summ_preprocessor import combine_intersecting_highlights
from Few_shot_experiments.utils import remove_spaces_and_punctuation
from src.inference.utils import _prepare_input, config_to_params_str
from src.train.highlight_to_question_preprocessor import HIGHLIGHT_SEP

import torch
import evaluate
import pandas as pd
import spacy
from tqdm import tqdm
import numpy as np
import logging
import json
from collections import defaultdict
from rouge_score import rouge_scorer, scoring

def get_results_path(config):
    return f"{config['highlights_representation_model_path']}/results/{config_to_params_str(config, '')}.json"


class FiCBacktracker:
    def __init__(self, similarity_calculator, dataset_type: str, preprocessor):
        self.similarity_calculator = similarity_calculator
        self.dataset_type = dataset_type
        self.preprocessor = preprocessor
        self.nlp = spacy.load("en_core_web_sm")
        
    def backtrace_fic_to_cluster(self, data, all_highlights, summarization_results, config = None):
        logging.info("backtracking FiC results to cluster")
        all_clusters = []
        
        assert len(data) == len(all_highlights)
        assert len(data) == len(summarization_results)
        for example, highlights, result in tqdm(zip(data, all_highlights, summarization_results)):
            clusters = self.handle_example(example, highlights, result)
            all_clusters.append(clusters)
        
        assert len(all_highlights) == len(all_clusters)
        assert len(all_highlights) == len(summarization_results)
        if config:
            if not config.get('filter_examples'):
                with open(get_results_path(config), "w") as f:
                    json.dump([{"clusters": clusters, "summarization_result": summarization_result} for highlights, clusters, summarization_result in list(zip(all_highlights, all_clusters, summarization_results))], f, indent=4)   
            
            return all_clusters
        else:
            return [{"clusters": clusters, "summarization_result": summarization_result} for highlights, clusters, summarization_result in list(zip(all_highlights, all_clusters, summarization_results))]
    

    def decompose_summary_to_sents(self, summary):
        return [x.text for x in list(self.nlp(summary).sents)]
    
    def handle_example(self, example, highlights, result):
        summary = result['predicted']
        
        # FiC generates sentences to be mapped to highlights
        if self.dataset_type == "fic":
            sentencized_summary = self.decompose_summary_to_sents(summary)
        # generative_clustering generates separators between highlights
        else:
            sentencized_summary = []
            for highlights_str in summary.split(self.preprocessor.special_tokens_constants['summary_sent_highlights_separator']):
                sentencized_summary.append(highlights_str)
        
        # Each disparate highlight can fall in a different cluster, so split based on the separator
        highlights_df = pd.DataFrame(highlights)
        highlights_df['docSpanText'] = highlights_df['docSpanText'].apply(lambda text: text.split(HIGHLIGHT_SEP))
        highlights_df = highlights_df.explode(['docSpanText', 'docSpanOffsets'])

        # Create a cluster for each summary sentence by running over all highlights and assigning them summary sentences
        clusters = self.similarity_calculator.create_clusters(highlights_df, sentencized_summary)

        # Split a highlight if its span is cross-sentence (necessary later for the attributable fusion)
        clusters = postprocess_clusters(clusters, example)
        
        return clusters, summary

    

class CalculateSimilarityByRouge():
    def __init__(self, highlight_sep_token: str) -> None:
        # self.metric = evaluate.load("rouge", seed=42)
        self.metric = rouge_scorer.RougeScorer(rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False, tokenizer=None)
        self.matching_strategy = 'highest_score'
        self.highlight_sep_token = highlight_sep_token
        
    def create_clusters(self, highlights_df, sentencized_summary):
        clusters = defaultdict(list)
        for _, highlight in highlights_df.iterrows():
            if self.matching_strategy == "highest_score":
                new_highlight_obj = self.find_highlight_most_similar_summary_sentence(highlight, sentencized_summary)
            else:
                raise NotImplementedError(f"Matching strategy {self.matching_strategy} not implemented")
            
            if new_highlight_obj is not None:
                summary_sent_idx = new_highlight_obj['scuSentIdx']
                                                        
                clusters[summary_sent_idx].append(new_highlight_obj)
                
        clusters = list(clusters.values())
        
        return clusters
        
    def similarity_score(self, highlight, sentencized_summary):
        # A single summary sentence can be comprised from multiple highlights in the generative clusters settings (not in the FiC settings)
        # We want to compare each highlight separately, but also keep track of the original summary sentence index
        flattened_summary = [(highlight, summary_sent_idx) for summary_sent_idx, summary_sent in enumerate(sentencized_summary) for highlight in summary_sent.split(self.highlight_sep_token)]
        
        # preprocess texts
        highlight = preprocess_text(highlight)
        flattened_summary_texts = [preprocess_text(x[0]) for x in flattened_summary]
        scores = []
        for summary_sent in flattened_summary_texts:
            scores.append(self.metric.score(target=highlight, prediction=summary_sent)['rougeL'].precision)
        
        # Flatten back highlights by keeping only the max score per summary index
        all_scores = {}
        assert len(scores) == len(flattened_summary)
        for score, x in zip(scores, flattened_summary):
            summary_sent_idx = x[1]
            all_scores[summary_sent_idx] = max(all_scores.get(summary_sent_idx, 0), score)
        
        final_scores = []
        for summary_sent_idx in range(len(sentencized_summary)):
            final_scores.append(all_scores[summary_sent_idx])
        
        return final_scores
            
    def find_highlight_most_similar_summary_sentence(self, highlight, sentencized_summary):
        scores = self.similarity_score(highlight['docSpanText'], sentencized_summary)
        scores = torch.tensor(scores)
        
        max_score = scores.max().item()
        if max_score > 0.0:
            more_than_one_match = (scores == max_score).sum() > 1
            # Decide by matching highlight sentence instead of span
            if more_than_one_match:
                scores = self.similarity_score(highlight['docSentText'], sentencized_summary)
                scores = torch.tensor(scores)  
                max_score = scores.max().item()
                more_than_one_match = (scores == max_score).sum() > 1
                
                if more_than_one_match:
                    logging.warning(f"More than one match for highlight '{highlight['docSpanText']}' in summary '{sentencized_summary}', even after comparing sentences. choosing first match")
            summary_sent_idx = scores.argmax().item()
                                
            return {
                "FiCBacktraceMaxScore": max_score,
                "documentFile": highlight['documentFile'],
                "docSpanText": highlight['docSpanText'],
                "docSentText": highlight['docSentText'],
                "sent_idx": highlight['sent_idx'],
                "docSentCharIdx": highlight['docSentCharIdx'],
                "docSpanOffsets": highlight['docSpanOffsets'],
                "scuSentCharIdx": None,  # we can calc this if necessary
                "scuSentence": sentencized_summary[summary_sent_idx],  # not used but might be nice for analysis
                "scuSentIdx": summary_sent_idx,  # used to sort the clusters
                "prefix": None  # we can calc this if necessary
            }
            
        return None



def preprocess_text(text):
    return ' '.join([word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')])


class CalculateSimilarityBySimilarityScore():
    def __init__(self, model_name, threshold, matching_strategy, matching_target) -> None:
        self.model_name = model_name
        if 'gte-large' in self.model_name:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        else:
            from AutoAIS.attribution_metrics.src.attribution_recall import AttributionRecall
            self.model = AttributionRecall(self.model_name)
        self.threshold = threshold
        self.matching_strategy = matching_strategy
        self.matching_target = matching_target

        
    def create_clusters(self, highlights_df, sentencized_summary):
        scores = self.calc_scores(highlights_df, sentencized_summary)
            
        clusters = defaultdict(list)
        for highlight_span_idx in range(highlights_df.shape[0]):
            if self.matching_strategy == "highest_score":
                max_score = scores[highlight_span_idx].max().item()
                max_score_idx = scores[highlight_span_idx].argmax().item()
                highlight_obj = highlights_df.iloc[highlight_span_idx]
                new_highlight_obj = create_new_highlight_obj(highlight_obj, sentencized_summary, max_score_idx, max_score)
                clusters[max_score_idx].append(new_highlight_obj)
            elif self.matching_strategy == "all_above_threshold":
                # Output the pairs with their score
                for summary_sent_idx in range(len(sentencized_summary)):
                    score = scores[highlight_span_idx][summary_sent_idx].item()
                    highlight_obj = highlights_df.iloc[highlight_span_idx]

                    logging.debug("Score: {:.4f} between highlight '{highlight_obj}' and summary sentence '{}'".format(
                        score, sentencized_summary[summary_sent_idx]
                    ))
                    
                    if score >= self.threshold:
                        new_highlight_obj = create_new_highlight_obj(highlight_obj, sentencized_summary, summary_sent_idx, score)
                        clusters[summary_sent_idx].append(new_highlight_obj)
            else:
                raise NotImplementedError(f"Matching strategy {self.matching_strategy} not implemented")
                
        clusters = list(clusters.values())
        
        return clusters
    
    def calc_scores(self, highlights_df, sentencized_summary):
        key_to_use = 'docSpanText' if self.matching_target == 'highlight' else 'docSentText'
        if 'gte-large' in self.model_name:
            sentencized_summary_encodings = self.model.encode(sentencized_summary)
            highlights_spans = highlights_df[key_to_use].tolist()
            highlights_spans_encodings = self.model.encode(highlights_spans)
            
            from sentence_transformers import util
            scores = util.cos_sim(highlights_spans_encodings, sentencized_summary_encodings)
        else:
            scores = []
            for highlight in highlights_df[key_to_use]:
                highlight_scores = []
                input_objs = []
                for summary_sent in sentencized_summary:
                    input_objs.append([{
                        "sentence": highlight,   # Hypothesis
                        "attribution": {  # Premise
                            "": summary_sent
                        }
                    }])
                score_objs = self.model.evaluate(input_objs)
                highlight_scores = score_objs['entail_cnt_scores']
                scores.append(np.array(highlight_scores))
                
        return scores
        

def postprocess_clusters(clusters, example):
    """
    Does the following post-processing
    * Split cross-sentence highlights
    * Sort clusters
    * Combine intersecting highlights
    * Filter highlights
    """
    
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for highlight_obj in cluster:
            new_highlight_objs = parse_cross_sent_highlight(highlight_obj, example)
            new_cluster.extend(new_highlight_objs)
        new_clusters.append(new_cluster)
    clusters = new_clusters
    
    # Sort clusters based on the summary sentences order
    clusters = sorted(clusters, key=lambda cluster: cluster[0]['scuSentIdx'])
    
    # Recombine highlights if they come from the same source sentence
    new_clusters = []
    for cluster in clusters:
        if len(cluster) == 0:
            logging.warning("Did not assign any highlights for this summary sentence")
        
        new_cluster = pd.DataFrame(cluster).groupby('documentFile').apply(lambda rows: combine_intersecting_highlights(rows, orig_text=None, combine_same_sent_highlights=True, group_by_summary_sentence=True)).to_dict('records')
        new_clusters.append(new_cluster)
    clusters = new_clusters

    # Filter highlights
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for highlight in cluster:
            if remove_spaces_and_punctuation(highlight['docSpanText']) != '':
                new_cluster.append(highlight)
                
        if len(new_cluster) > 0:
            new_clusters.append(new_cluster)
    clusters = new_clusters

    return clusters

def create_new_highlight_obj(highlight, sentencized_summary, summary_sent_idx, score):
    return {
        "FiCBacktraceMaxScore": score,
        "documentFile": highlight['documentFile'],
        "docSpanText": highlight['docSpanText'],
        "docSentText": highlight['docSentText'],
        "sent_idx": highlight['sent_idx'],
        "docSentCharIdx": highlight['docSentCharIdx'],
        "docSpanOffsets": highlight['docSpanOffsets'],
        "scuSentCharIdx": None,  # we can calc this if necessary
        "scuSentence": sentencized_summary[summary_sent_idx],  # not used but might be nice for analysis
        "scuSentIdx": summary_sent_idx,  # used to sort the clusters
        "prefix": None  # we can calc this if necessary
    }


def parse_cross_sent_highlight(highlight_obj, example):
    """
    Some highlights are cross-sentence, so we want to split them into separate rows.
    For example, highlight_obj['docSpanOffsets'] = [[5170,5217]], but based on document['docSentCharIdxToSentIdx'] = [..., 5170, 5204, ...], we can see that the highlight is cross-sentence.
    """

    documents = example['documents']
    document = [d for d in documents if highlight_obj['documentFile'] == d['documentFile']]
    assert len(document) == 1
    document = document[0]

    new_highlight_objs = []
    start, end = highlight_obj['docSpanOffsets']
    docSentCharIdxToSentIdx = document.get('docSentCharIdxToSentIdx', get_doc_sent_char_idx_to_sent_idx(document['documentText'], document['rawDocumentText']))
    start_sent_idx = docSentCharIdxToSentIdx.index([x for x in docSentCharIdxToSentIdx if x <= start][-1])
    end_sent_idx = docSentCharIdxToSentIdx.index([x for x in docSentCharIdxToSentIdx if x <= end][-1])

    for sent_idx in range(start_sent_idx, end_sent_idx+1):
        if sent_idx == start_sent_idx:
            curr_start = start
        else:
            curr_start = docSentCharIdxToSentIdx[sent_idx]
        
        if sent_idx == end_sent_idx:
            curr_end = end
        else:
            curr_end = docSentCharIdxToSentIdx[sent_idx+1]
            
        new_span_text = document['rawDocumentText'][curr_start:curr_end]
        doc_sent_text = document['documentText'][sent_idx]
        
        assert new_span_text.strip() in highlight_obj['docSpanText']
        assert new_span_text.strip() in doc_sent_text
        
        new_highlight_objs.append({
            **highlight_obj,
            'docSpanOffsets': [curr_start, curr_end],
            'docSpanText': new_span_text,
            'docSentText': doc_sent_text,
            'docSentCharIdx': docSentCharIdxToSentIdx[sent_idx],
            'sent_idx': sent_idx
        })
    
    
    return new_highlight_objs
    
def get_doc_sent_char_idx_to_sent_idx(document_sents, document_raw_text):
    docSentCharIdxToSentIdx = []
    running_idx = 0
    for sent_id, sent in enumerate(document_sents):
        truncated_sent_idx = document_raw_text.index(sent)
        start = truncated_sent_idx + running_idx

        docSentCharIdxToSentIdx.append(start)

        # Remove used text to handle cases where there is duplications
        running_idx += len(sent)
        document_raw_text = document_raw_text[len(sent):]

    return docSentCharIdxToSentIdx

