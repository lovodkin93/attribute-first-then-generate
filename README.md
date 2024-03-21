# attribute-first-then-generate

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

## Fine-tuned models

Scripts for fine-tuning models and running fine-tuned models are in the directory `fine-tuned`.

### Training

`train.py` is used to train all fine-tuned models.


### Inference

`inference.py` is used for the inference of First-attribute models.

`inference_e2e.py` is used for the inference of the vanilla baseline Primera model.