import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock
import torch
from Few_shot_experiments.run_content_selection_MDS import adapt_highlights_to_doc_alignments, get_set_of_highlights_in_context_content_selection
from src.train.alignment_model import AlignmentModelConfig
from src.train.attributable_fusion_preprocessor import AttributableFusionPreprocessor
from src.train.highlight_to_question_preprocessor import HighlightToQuestionPreprocessor
from src.train.highlight_to_question_summ_preprocessor import HighlightToQuestionSummPreprocessor
from src.train.highlight_detection_preprocessor import HighlightDetectionPreprocessor
from src.train.highlights_detection_postprocessor import HighlightsDetectionPostprocessor
from src.train.led_model_encoder import LEDEncoderForTokenClassification, LEDModelEncoderOnly
from src.train.predictions_analyzer import PredictionsAnalyzer
from src.train.generation_e2e_preprocessor import GenerationE2EPreprocessor
from src.train.utils import prepare_config_for_hf, save_minified_config

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    # NEW
    dataset_type: Optional[str] = field(
        default=None, metadata={"help": "Dataset type"}
    )
    # NEW
    task_type: Optional[str] = field(
        default=None, metadata={"help": "Task type (QA, MDS)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    # NEW
    preprocess_context_window: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of sentences to take from either side of the highlight"
            )
        },
    )
    # NEW
    preprocess_only_before: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Should we take sentences from both sides of the highlight or only one side"
            )
        },
    )
    # NEW
    is_classification_task: Optional[bool] = field(
        default=False
    )
    # NEW from original script
    do_hyperparameter_tuning: bool = field(default=False)


    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def wandb_hp_space_wrapper(training_args):
    def wandb_hp_space(trial):
        return {
            'method': 'random',
            'name': 'sweep',
            'metric': {
                'name': training_args.metric_for_best_model,
                'goal': 'maximize'
            },
            'parameters': {
                "gradient_accumulation_steps": {'values': [1]},
                # "per_device_train_batch_size": {'values': [2, 4, 8]},
                'num_train_epochs': {'values': [3]},
                'learning_rate': {'max': 5e-4, 'min': 5e-8},
                # 'learning_rate': {'values': [1e-8, 5e-7, 1e-7, 5e-6, 1e-6]},
                # 'warmup_steps': {'values': [0, 100]},
                'warmup_steps': {'max': 300, 'min': 0},
                'weight_decay': {
                    'values': [0.0, 0.2, 0.5]
                },
            }
        }

    return wandb_hp_space



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # NEW - support classification
    is_classification_task = 'w_encoder' in sys.argv[1]
    arguments_class = Seq2SeqTrainingArguments if not is_classification_task else TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, arguments_class))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # NEW - remove telemetry
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if not data_args.is_classification_task:
        # NEW from original script (sweep requires a model_init since it initializes the model every sweep. moved everything related to its init to here)
        def model_init_wrapper(tokenizer):
            def model_init(trial):
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    token=model_args.token,
                    trust_remote_code=model_args.trust_remote_code,
                )
                
                model_update(model, tokenizer)
                
                return model
            return model_init
        
        # NEW from original script (everything related to the model update so we can call it also later)
        def model_update(model, tokenizer):
            # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
            # on a small vocab and want a smaller embedding size, remove this test.
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
                if isinstance(tokenizer, MBartTokenizer):
                    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
                else:
                    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

            if model.config.decoder_start_token_id is None:
                raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

            if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < data_args.max_source_length
            ):
                if model_args.resize_position_embeddings is None:
                    logger.warning(
                        "Increasing the model's number of position embedding vectors from"
                        f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
                    )
                    model.resize_position_embeddings(data_args.max_source_length)
                elif model_args.resize_position_embeddings:
                    model.resize_position_embeddings(data_args.max_source_length)
                else:
                    raise ValueError(
                        f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                        f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                        f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                        " model's position encodings by passing `--resize_position_embeddings`."
                    )

                    
        model_init = model_init_wrapper(tokenizer)
        model = model_init(None)
    else:
        config.num_labels = 2
        
        # NEW from original script (sweep requires a model_init since it initializes the model every sweep)
        def model_init_wrapper(tokenizer):
            def model_init(trial):
                return LEDEncoderForTokenClassification.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    token=model_args.token,
                    trust_remote_code=model_args.trust_remote_code,
                    device=training_args.device
                )
            return model_init
                  
        model_init = model_init_wrapper(tokenizer)
        model = model_init(None)
        
    # NEW from original script
    is_t5_model = model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ] or model.config.model_type == 't5'  # Necessary when loading model from directory

    # NEW from original script
    if data_args.dataset_type == 'attributable_fusion' and False:
        # TODO: AlignmentModel needs to have inputs_ids and attention_mask in signature
        from src.train.alignment_model import AlignmentModel

        AlignmentModelConfig(architectures=config)
        model = AlignmentModel(config, seq2seq_model=model)

    # NEW from original script
    if data_args.dataset_type == 'highlight_detection':
        from src.train.highlight_detection_preprocessor import get_special_tokens_constants
    elif data_args.dataset_type == 'highlight_to_question__quote_sum':
        from src.train.highlight_to_question_preprocessor import get_special_tokens_constants
    elif data_args.dataset_type == 'highlight_to_question__summ':
        from src.train.highlight_to_question_summ_preprocessor import get_special_tokens_constants
    elif data_args.dataset_type in ['attributable_fusion', 'fic', 'generative_clustering', 'generative_highlight_n_cluster']:
        from src.train.attributable_fusion_preprocessor import get_special_tokens_constants
    elif data_args.dataset_type == 'generation_e2e':
        from src.train.generation_e2e_preprocessor import get_special_tokens_constants
    else:
        raise ValueError(f"Unknown dataset type: {data_args.dataset_type}")        
    special_tokens_constants = get_special_tokens_constants(is_t5_model)
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens_constants.values())})
    # NEW from original script - call model_update instead of the code that was here before
    model_update(model, tokenizer)

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    # NEW from original script
    if data_args.dataset_type == 'highlight_detection':
        is_primera = model.config.model_type == 'led'
        preprocessor = HighlightDetectionPreprocessor(
            special_tokens_constants,
            tokenizer=tokenizer,
            data_args=data_args,
            device=training_args.device,
            max_target_length=max_target_length,
            padding=padding,
            add_global_attention=is_primera
        )
        
        preprocess_function = lambda examples: preprocessor.preprocess_function(
            examples,
            is_training=True,
            is_classification_task=data_args.is_classification_task,
            should_extract_targets=True
        )
        
        key = "validation"
        # key = "test"
        raw_dataset_to_use_for_postprocessing = raw_datasets[key]
        postprocessor = HighlightsDetectionPostprocessor(tokenizer=tokenizer, special_tokens_constants=special_tokens_constants, raw_dataset=raw_dataset_to_use_for_postprocessing)
    elif data_args.dataset_type == 'highlight_to_question__quote_sum':
        preprocessor = HighlightToQuestionPreprocessor(special_tokens_constants, tokenizer=tokenizer, data_args=data_args, max_target_length=max_target_length, padding=padding)
        
        preprocess_function = preprocessor.preprocess_function
    elif data_args.dataset_type == 'highlight_to_question__summ':
        preprocessor = HighlightToQuestionSummPreprocessor(special_tokens_constants, tokenizer=tokenizer, data_args=data_args, max_target_length=max_target_length, padding=padding)
        
        preprocess_function = preprocessor.preprocess_function
    elif data_args.dataset_type in ['attributable_fusion', 'fic', 'generative_clustering', 'generative_highlight_n_cluster']:
        group_by_summary_sentence = data_args.dataset_type == "attributable_fusion"
        
        preprocessor = AttributableFusionPreprocessor(
            special_tokens_constants,
            tokenizer=tokenizer,
            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
            max_source_length=data_args.max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            context_window=data_args.preprocess_context_window,
            only_before=data_args.preprocess_only_before,
            device=training_args.device,
            group_by_summary_sentence=group_by_summary_sentence,
            dataset_type=data_args.dataset_type
        )
        preprocess_function = lambda examples: preprocessor.preprocess_function(examples, is_training=True, should_extract_targets=True)
    elif data_args.dataset_type == 'generation_e2e':
        preprocessor = GenerationE2EPreprocessor(
            special_tokens_constants,
            tokenizer=tokenizer,
            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
            max_source_length=data_args.max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            device=training_args.device,
        )
        preprocess_function = lambda examples: preprocessor.preprocess_function(examples, is_training=True, should_extract_targets=True)
    else:
        raise ValueError(f"Unknown dataset type: {data_args.dataset_type}")

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,  # NEW from original_script
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,  # NEW from original_script
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            # NEW from original_script
            prep_predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    
    # NEW
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if not data_args.is_classification_task:
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
            data_collator = DataCollatorForTokenClassification(
                tokenizer,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None
            )

    # NEW - add seed to rouge
    if not data_args.is_classification_task:
        metric = evaluate.load("rouge", seed=42)
    else:
        metric = datasets.load_metric("f1")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        if data_args.dataset_type != 'highlight_detection':
            # Replace -100s used for padding as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result
        elif not data_args.is_classification_task:
            # TODO: Receive metric_key_prefix from previous trainer and use it here to decide if to take the raw_datasets
            # key = "validation"
            # key = "test"
            all_preds_set_of_highlights_in_context, all_labels_set_of_highlights_in_context = postprocessor.extract_prediction(preds, labels, raw_examples=None)

            # Extract all predicted words indices
            all_results = []
            for preds_set_of_highlights_in_context, labels_set_of_highlights_in_context in zip(all_preds_set_of_highlights_in_context, all_labels_set_of_highlights_in_context):
                preds_indices = set([x for z in preds_set_of_highlights_in_context for y in z['docSpanOffsets'] for x in range(*y)])
                labels_indices = set([x for z in labels_set_of_highlights_in_context for y in z['docSpanOffsets'] for x in range(*y)])
                
                # Calc IOU
                intersection_indices = preds_indices.intersection(labels_indices)
                union_indices = preds_indices.union(labels_indices)
                
                result = len(intersection_indices) / len(union_indices)
                all_results.append(result)

            mean_iou = np.array(all_results).mean()
            mean_iou = round(mean_iou * 100, 4)
            return {
                "mean_iou": result
            }
        else:            
            preds = np.argmax(preds, axis=2)
            
            flattened_preds = torch.tensor(preds).view(-1)
            flattened_labels = torch.tensor(labels).view(-1)
            non_pad_indices = flattened_labels != -100
            
            preds = flattened_preds[non_pad_indices]
            labels = flattened_labels[non_pad_indices]
            
            result = metric.compute(predictions=preds, references=labels)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            return result

    # NEW - only if not classification
    if not data_args.is_classification_task:
        # Override the decoding parameters of Seq2SeqTrainer
        training_args.generation_max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_target_length
        )
        training_args.generation_num_beams = (
            data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        )

    # Initialize our Trainer
    if not data_args.is_classification_task:        
        trainer = Seq2SeqTrainer(
            model=None if data_args.do_hyperparameter_tuning else model,
            model_init=model_init if data_args.do_hyperparameter_tuning else None,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None
        )
    else:
        trainer = Trainer(
            model=None if data_args.do_hyperparameter_tuning else model,
            model_init=model_init if data_args.do_hyperparameter_tuning else None,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
    
    # NEW - save data_args
    with open(f"{trainer.args.output_dir}/data_args.json", 'w') as f:
        json.dump(data_args.__dict__, f, indent=4)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            

        # NEW from original script (hyperparameter search)
        if data_args.do_hyperparameter_tuning:
            def compute_objective(res):
                return res[training_args.metric_for_best_model]

            hyperparameters_result = trainer.hyperparameter_search(
                    direction="maximize",
                    backend="wandb" ,
                    hp_space=wandb_hp_space_wrapper(training_args),
                    n_trials=10,
                    compute_objective=compute_objective
                )
                
            logger.info(f"Best trial : {hyperparameters_result.hyperparameters}")
            return 0

        else:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(prep_predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        # NEW - change from predict to include file name
        file_path_id = '.'.join(data_args.test_file.replace('/', '_').split('.')[:-1])
        trainer.log_metrics(f"predict_{file_path_id}", metrics)
        trainer.save_metrics(f"predict_{file_path_id}", metrics)

        if trainer.is_world_process_zero():
            if data_args.is_classification_task:
                predictions = predict_results.predictions
                df = pd.DataFrame(predict_dataset.to_dict())
                predictions_analyzer = PredictionsAnalyzer(tokenizer, preprocessor, output_dir=training_args.output_dir, is_classification_task=data_args.is_classification_task)
                predictions_analyzer.write_predictions_to_file(predictions=predict_results.predictions, dataset=prep_predict_dataset, df=df, data_file_path=data_args.test_file)
            elif training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding='utf-8') as writer:
                    writer.write("\n".join(predictions))

                # NEW from original script
                df = pd.DataFrame(predict_dataset.to_dict())
                predictions_analyzer = PredictionsAnalyzer(tokenizer, preprocessor, output_dir=training_args.output_dir, is_classification_task=data_args.is_classification_task)
                predictions_analyzer.write_predictions_to_file(predictions=predict_results.predictions, dataset=prep_predict_dataset, df=df, data_file_path=data_args.test_file)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


# NEW
def update_output_dir(config):
    unique_dir_name = f"{config['task_type']}__{config['model_name_or_path'].replace('/', '_')}"
    config['output_dir'] = f"{config['output_dir']}/{unique_dir_name}"


if __name__ == "__main__":
    
    config, config_file_path = prepare_config_for_hf()
    update_output_dir(config)

    save_minified_config(config, config_file_path)


    main()
