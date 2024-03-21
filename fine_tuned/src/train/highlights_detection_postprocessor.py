import numpy as np
import spacy


class HighlightsDetectionPostprocessor:
    def __init__(self, tokenizer, special_tokens_constants, raw_dataset):
        self.tokenizer = tokenizer
        self.special_tokens_constants = special_tokens_constants
        self.raw_dataset = raw_dataset
        
        self.highlights_separator_idx = self.tokenizer.vocab[self.special_tokens_constants['highlights_separator']]
        self.docs_sep_idx = self.tokenizer.vocab[self.special_tokens_constants['documents_separator']]
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_prediction(self, preds, labels, raw_examples):
        """
        The prediction is a set of extractive highlights, we need to backtrace these back them to spans in the original docment so we can evaluate if they were highlighted or not
        """
        
        if raw_examples is None:
            raw_examples = self.raw_dataset
        
        
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = [[x[0] for x in self.decode_pred(pred)] for pred in preds]
        
        decoded_labels = None
        if labels is not None:
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = [[x[0] for x in self.decode_pred(label)] for label in labels]
        
        return self.find_spans_in_docs(decoded_preds, raw_examples, decoded_labels)

    def find_spans_in_docs(self, decoded_preds, raw_examples, decoded_labels):
        if decoded_labels is None:
            decoded_labels = [None] * len(decoded_preds)
        
        # convert predictions and labels to spans in doc
        all_preds_set_of_highlights_in_context, all_labels_set_of_highlights_in_context = [], []
        assert len(decoded_preds) == len(raw_examples)
        assert len(decoded_preds) == len(decoded_labels)
        for decoded_pred, decoded_label, raw_example in zip(decoded_preds, decoded_labels, raw_examples):
            docs = raw_example['documents']
            preds_set_of_highlights_in_context = self.find_offsets_to_list_of_spans(decoded_pred, docs)
            all_preds_set_of_highlights_in_context.append(preds_set_of_highlights_in_context)

            if decoded_label is not None:
                labels_set_of_highlights_in_context = self.find_offsets_to_list_of_spans(decoded_label, docs)
                all_labels_set_of_highlights_in_context.append(labels_set_of_highlights_in_context)
            
        return all_preds_set_of_highlights_in_context, all_labels_set_of_highlights_in_context


    def find_offsets_to_list_of_spans(self, decoded, docs):
        from Few_shot_experiments.run_content_selection_MDS import adapt_highlights_to_doc_alignments, get_set_of_highlights_in_context_content_selection

        set_of_highlights_in_context = []

        # Find document for each span
        doc_texts_dict = {doc['documentFile']: doc['rawDocumentText'] for doc in docs}
        doc_texts_dict['fake'] = ''
        salience_dict = {"fake": decoded}
        doc_id_to_label = adapt_highlights_to_doc_alignments(doc_texts_dict, salience_dict, skip_failed=True)
        
        # Find indices for each document
        for doc in docs:
            doc_highlights = doc_id_to_label.get(doc['documentFile'], [])
            highlights = get_set_of_highlights_in_context_content_selection(doc['documentFile'], doc['rawDocumentText'], doc_highlights, self.nlp, doc['documentText'])
            for highlight in highlights:
                highlight['docSentCharIdx'] = int(highlight['docSentCharIdx'])
            set_of_highlights_in_context.extend(highlights)
            
        return set_of_highlights_in_context
    
    
    def decode_pred(self, pred):
        """
        Given a prediction such as
        "</s><s><s>abc<highlight-sep>def<doc-sep>ghi"
        returns:
        ["abc", "def", "ghi"]
        
        Complicated because there are different types of special tokens
        """
        
        def read_highlight(i, highlights_separation_indices):
            highlight_start_offset = 0
            highlight_end_offset = len(pred)
            if len(highlights_separation_indices) > 0:
                if i > 0:
                    highlight_start_offset = highlights_separation_indices[i-1] + 1
                if i < len(highlights_separation_indices):
                    highlight_end_offset = highlights_separation_indices[i]
            pred_highlight = pred[highlight_start_offset:highlight_end_offset]
            decoded_pred_highlight = self.tokenizer.decode(pred_highlight, skip_special_tokens=True)
            return (decoded_pred_highlight, pred_highlight)
        
        pred_highlights = []
        
        highlights_separation_indices = np.where(np.isin(pred, [self.highlights_separator_idx, self.docs_sep_idx]))[0]
        for i in range(len(highlights_separation_indices)):
            pred_highlights.append(read_highlight(i, highlights_separation_indices))
        
        # Read also last one / the only one
        pred_highlights.append(read_highlight(len(highlights_separation_indices), highlights_separation_indices))

        return pred_highlights
