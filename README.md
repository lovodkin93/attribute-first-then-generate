# Attribute First, then Generate

<p align=left>
    <img src="./First-attribute-then-generate architecture.jpg" width="60%" height="60%"  alt="taxonomy"/>
</p>

# Data format

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
## Few-shot experiments
Scripts for few-shot experiments are in the directory `few_shot_experiments`.

For the different subtasks (except the iterative sentence generation), run:
```
python run_script.py --config-file configs/<SPLIT>/<SETTING>/<SUBTASK>.json
```
where:
* **<SPLIT>** - dev or test
* **<SETTING>** - MDS or LFQA
* **<SUBTASK>** - either one of: content_selection, clustering, fusion_in_context, e2e_only_setting, or ALCE

For the iterative sentence generation, run:
```
python run_iterative_sent_gen.py --config-file configs/<SPLIT>/<SETTING>/iterative_sentence_generation.json
```
where **<SPLIT>** and **<SETTING>** are the same as before.

When running each of the components of the pipeline, you will find in the outdir directory a file called `pipeline_format_results.json`.
This file should be passed to the next component in the pipeline as input, by adding the following flag, in addition to `config_file`:
```
--indir-alignments /path/to/pipeline_format_results.json
```

## Fine-tuned models

Scripts for fine-tuning models and running fine-tuned models are in the directory `fine-tuned`.

### Training

`train.py` is used to train all fine-tuned models.


### Inference

`inference.py` is used for the inference of First-attribute models.

`inference_e2e.py` is used for the inference of the vanilla baseline Primera model.

## Few-shot in-context learning variants
Code coming soon!

# Citation
If you use this in your work, please cite:
```
@misc{slobodkin2024attribute,
      title={Attribute First, then Generate: Locally-attributable Grounded Text Generation}, 
      author={Aviv Slobodkin and Eran Hirsch and Arie Cattan and Tal Schuster and Ido Dagan},
      year={2024},
      eprint={2403.17104},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
