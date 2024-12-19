# Attribute First, then Generate

<p align=left>
    <img src="./First-attribute-then-generate architecture.jpg" width="60%" height="60%"  alt="taxonomy"/>
</p>

# Data Format

```python
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Example:
    unique_id: str  # multi-news topic
    topic: str  # multi-news topic
    documents: List[Document]
    set_of_highlights_in_context: List[Highlight]
    response: str = List[str]
    question: str = Optional[str]


@dataclass
class Document:
    documentFile: str  # document ID
    rawDocumentText: str  # document text as a single string
    documentText: List[str]  # document text split to sentences
    docSentCharIdxToSentIdx: List[int]  # char idx of each sentence
    documentTitle: Optional[str] = None
    documentUrl: Optional[str] = None


@dataclass
class Highlight:
    documentFile: str  # document ID
    docSentText: str  # document sentence
    sent_idx: int  # index of the document sentence
    docSentCharIdx: int  # character index of the document sentence
    docSpanText: str  # span of the document sentence
    docSpanOffsets: str  # offset of the document sentence span
    scuSentence: str  # summary sentence
    scuSentCharIdx: float  # character index of the summary sentence
    summarySpanText: str  # span of the summary sentence
    summarySpanOffsets: str  # offset of the summary sentence span
    prefix: str  # previous summary sentences
    
```
## Few-shot Experiments

Scripts for few-shot experiments are in the directory `few_shot_experiments`.

### Separate Subtasks
For the different subtasks (except the iterative sentence generation), run:
```
python run_script.py --config-file configs/<SPLIT>/<SETTING>/<SUBTASK>.json
```
where:
* **SPLIT** - dev or test
* **SETTING** - MDS or LFQA
* **SUBTASK** - either one of: content_selection, clustering, fusion_in_context, e2e_only_setting, or ALCE

For the iterative sentence generation, run:
```
python run_iterative_sentence_generation.py --config-file configs/<SPLIT>/<SETTING>/iterative_sentence_generation.json
```
where **SPLIT** and **SETTING** are the same as before.

When running each of the components of the pipeline, you will find in the outdir directory a file called `pipeline_format_results.json`. \
This file should be passed to the next component in the pipeline as input, by adding the following flag, in addition to `config_file`:
```
--indir-alignments /path/to/pipeline_format_results.json
```
So for example, when running the full CoT variant, after running the *content selection* component, you should run:
```
python run_script.py --config-file configs/<SPLIT>/<SETTING>/fusion_in_context.json --indir-alignments /path/to/content_selection pipeline_format_results.json
```

### Full Pipelines
For the full pipelines, run:
```
python run_full_pipeline.py --config-file configs/<SPLIT>/<SETTING>/<PIPELINE_TYPE>.json
```
where **PIPELINE_TYPE** is one of the following:
* `full_pipeline` - the pipeline consisting of three subtasks - content selection, clustering, and iterative sentence generation.
* `full_CoT_pipeline` - the pipeline consisting of two subtasks - content selection and CoT-like sentence fusion.

## Fine-tuned Models

Scripts for fine-tuning models and running fine-tuned models are in the directory `fine-tuned`.

### Training

`train.py` is used to train all fine-tuned models.


### Inference

`inference.py` is used for the inference of First-attribute models.

`inference_e2e.py` is used for the inference of the vanilla baseline Primera model.

# Citation
If you use this in your work, please cite:
```
@inproceedings{slobodkin-etal-2024-attribute,
    title = "Attribute First, then Generate: Locally-attributable Grounded Text Generation",
    author = "Slobodkin, Aviv  and
      Hirsch, Eran  and
      Cattan, Arie  and
      Schuster, Tal  and
      Dagan, Ido",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.182",
    doi = "10.18653/v1/2024.acl-long.182",
    pages = "3309--3344",
}
```
