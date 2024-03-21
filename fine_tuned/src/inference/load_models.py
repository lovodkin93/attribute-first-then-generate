import json
import logging
import torch
import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.train.highlights_detection_postprocessor import HighlightsDetectionPostprocessor
from src.train.led_model_encoder import LEDEncoderForTokenClassification

def load_highlights_detection_model(config, model_path):
    from src.train.highlight_detection_preprocessor import HighlightDetectionPreprocessor, get_special_tokens_constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    is_primera = 'flan-t5' not in model_path
    if is_primera:
        from transformers import LEDForConditionalGeneration
        model = LEDForConditionalGeneration.from_pretrained(model_path)
    elif 'flan-t5' in model_path:
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load preprocessor
    with open(f"{model_path}/data_args.json") as f:
        data_args = json.loads(f.read())
    
    is_t5_model = model.config.model_type == 't5'
    special_tokens_constants = get_special_tokens_constants(is_t5_model=is_t5_model)
    highlights_detection_preprocessor = HighlightDetectionPreprocessor(
        special_tokens_constants=special_tokens_constants,
        tokenizer=tokenizer,
        data_args=data_args,
        max_target_length=data_args['max_target_length'],
        padding=False,
        device=device,
        add_global_attention=is_primera
    )
    
    postprocessor = HighlightsDetectionPostprocessor(
            tokenizer=tokenizer,
            special_tokens_constants=special_tokens_constants,
            raw_dataset=None
        )
    
    return model, tokenizer, device, highlights_detection_preprocessor, postprocessor


def load_highlights_detection_model_w_encoder(config, model_path):
    from src.train.highlight_detection_preprocessor import HighlightDetectionPreprocessor, get_special_tokens_constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    is_primera = True
    if is_primera:
        model = LEDEncoderForTokenClassification.from_pretrained(model_path, device=device)
    else:
        model = AutoModel.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load preprocessor
    with open(f"{model_path}/data_args.json") as f:
        data_args = json.loads(f.read())
    
    is_t5_model = model.config.model_type == 't5'
    highlights_detection_preprocessor = HighlightDetectionPreprocessor(
        special_tokens_constants=get_special_tokens_constants(is_t5_model=is_t5_model),
        tokenizer=tokenizer,
        data_args=data_args,
        max_target_length=data_args['max_target_length'],
        padding=False,
        device=device
    )
    
    return model, tokenizer, device, highlights_detection_preprocessor


def load_highlights_representation_model(config):
    model_path = config['highlights_representation_model_path']
    model = SentenceTransformer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def load_fusion_model(config, model_path, group_by_summary_sentence: bool, dataset_type: str):
    from src.train.attributable_fusion_preprocessor import AttributableFusionPreprocessor, get_special_tokens_constants
    
    # Load model
    is_primera = True
    if is_primera:
        from transformers import LEDForConditionalGeneration
        model = LEDForConditionalGeneration.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load preprocessor
    with open(f"{model_path}/data_args.json") as f:
        data_args = json.loads(f.read())
    
    is_t5_model = model.config.model_type == 't5'
    attributable_fusion_preprocessor = AttributableFusionPreprocessor(
        special_tokens_constants=get_special_tokens_constants(is_t5_model=is_t5_model),
        tokenizer=tokenizer,
        ignore_pad_token_for_loss=None,
        max_source_length=data_args['max_source_length'],
        max_target_length=data_args['max_target_length'],
        padding=False,
        context_window=data_args['preprocess_context_window'],
        only_before=data_args['preprocess_only_before'],
        device=device,
        group_by_summary_sentence=group_by_summary_sentence,
        dataset_type=dataset_type
    )
    
    return model, tokenizer, device, attributable_fusion_preprocessor



def load_attributable_fusion_model(config):
    logging.info('loading model for fusion')
    
    model_path = config['attributable_fusion_model_path']
    # We only send a single cluster
    return load_fusion_model(config, model_path, group_by_summary_sentence=True, dataset_type='attributable_fusion')


def load_fic_model(config):
    logging.info('loading model for FiC')
    
    model_path = config["highlights_representation_model_path"]
    return load_fusion_model(config, model_path, group_by_summary_sentence=False, dataset_type='fic') 


def load_generative_clustering_model(config):
    logging.info('loading model for generative clustering')
    
    model_path = config['highlights_representation_model_path']
    
    return load_fusion_model(config, model_path, group_by_summary_sentence=False, dataset_type='generative_clustering')


def load_generative_highlight_n_cluster_model(config):
    logging.info('loading model for generative highlight & cluster')
    
    model_path = config['highlights_representation_model_path']
    
    model, tokenizer, device, attributable_fusion_preprocessor = load_fusion_model(config, model_path, group_by_summary_sentence=False, dataset_type='generative_highlight_n_cluster')

    postprocessor = HighlightsDetectionPostprocessor(
            tokenizer=tokenizer,
            special_tokens_constants=attributable_fusion_preprocessor.special_tokens_constants,
            raw_dataset=None
        )
    
    return model, tokenizer, device, attributable_fusion_preprocessor, postprocessor




def load_e2e_model(config):
    logging.info('loading model for e2e')
    
    model_path = config['e2e_model_path']

    from src.train.generation_e2e_preprocessor import GenerationE2EPreprocessor, get_special_tokens_constants
    
    # Load model
    is_primera = True
    if is_primera:
        from transformers import LEDForConditionalGeneration
        model = LEDForConditionalGeneration.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load preprocessor
    data_args_path = f"{model_path}/data_args.json"
    if not os.path.exists(data_args_path):
        data_args_path = f"{config['data_path']}/data_args.json"
    with open(data_args_path) as f:
        data_args = json.loads(f.read())

    is_t5_model = model.config.model_type == 't5'
    preprocessor = GenerationE2EPreprocessor(
        special_tokens_constants=get_special_tokens_constants(is_t5_model=is_t5_model),
        tokenizer=tokenizer,
        ignore_pad_token_for_loss=None,
        max_source_length=data_args['max_source_length'],
        max_target_length=data_args['max_target_length'],
        padding=False,
        device=device
    )

    return model, tokenizer, device, preprocessor
