# Hengam: An Adversarially Trained Transformer for Persian Temporal Tagging

## HuggingFace
- [HengamTrans Space](https://huggingface.co/spaces/kargaranamir/Hengam)
- [HengamTagger (Parstdex) Space](https://huggingface.co/spaces/kargaranamir/parstdex)
- [HengamTrans Models (ModelCard)](https://huggingface.co/kargaranamir/Hengam)
- [HengamCorpus (Dataset)](https://huggingface.co/datasets/kargaranamir/HengamCorpus)


## Code (Software)

### HengamTagger
The HengamTagger is distributed as [Parstdex](https://github.com/kargaranamir/parstdex) package (will be continuously updated) and is available via [pip](https://pypi.org/project/parstdex). 

### HengamTransformer

You can use this model directly downloading the utils and requirements files and installing requirements:

```python
! wget https://huggingface.co/spaces/kargaranamir/Hengam/raw/main/utils.py
! wget https://huggingface.co/spaces/kargaranamir/Hengam/raw/main/requirements.txt
! pip install -r requirements.txt
```

and downloading the models HengamTransA.pth or HengamTransW.pth and building ner pipline:

```python
import torch
from huggingface_hub import hf_hub_download
from utils import *

# HengamTransW = hf_hub_download(repo_id="kargaranamir/Hengam", filename="HengamTransW.pth")
HengamTransA = hf_hub_download(repo_id="kargaranamir/Hengam", filename="HengamTransA.pth")
```

```python
# ner = NER(model_path=HengamTransW, tags=['B-TIM', 'I-TIM', 'B-DAT', 'I-DAT', 'O'])
ner = NER(model_path=HengamTransA, tags=['B-TIM', 'I-TIM', 'B-DAT', 'I-DAT', 'O'])
ner('.سلام من و دوستم ساعت ۸ صبح روز سه شنبه رفتیم دوشنبه بازار ')
>>
[{'Text': 'ساعت', 'Tag': 'B-TIM', 'Start': 17, 'End': 21},
 {'Text': '۸', 'Tag': 'I-TIM', 'Start': 22, 'End': 23},
 {'Text': 'صبح', 'Tag': 'I-TIM', 'Start': 24, 'End': 27},
 {'Text': 'روز', 'Tag': 'I-TIM', 'Start': 28, 'End': 31},
 {'Text': 'سه', 'Tag': 'B-DAT', 'Start': 32, 'End': 34},
 {'Text': 'شنبه', 'Tag': 'I-DAT', 'Start': 35, 'End': 39}]
```



Alos, in this github 4 different notebooks are provided to train and load the model. Click on the hyperlinks to open each in Google Colab.

- Inference and Test 
  - [Inference_HengamTransW.ipynb](https://colab.research.google.com/github/kargaranamir/hengam/blob/main/code/Inference_HengamTransW.ipynb): In this notebook, [HengamTransW.pth](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransW.pth) is downloaded from [Hengam HuggingFace model card](https://huggingface.co/kargaranamir/Hengam), and an inference is provided in the last cells.
  - [Inference_HengamTransA.ipynb](https://colab.research.google.com/github/kargaranamir/hengam/blob/main/code/Inference_HengamTransA.ipynb): In this notebook, [HengamTransA.pth](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransA.pth) is downloaded from [Hengam HuggingFace model card](https://huggingface.co/kargaranamir/Hengam), and an inference is provided in the last cells.
- Train
  - [Train_HengamTransW.ipynb](https://colab.research.google.com/github/kargaranamir/hengam/blob/main/code/Train_HengamTransW.ipynb): This notebook uses the [HengamCorpus](https://huggingface.co/datasets/kargaranamir/HengamCorpus) dataset uploaded on HuggingFace. Afterwards, the HengamTransW will be trained.
  - [Train_HengamTransA.ipynb](https://colab.research.google.com/github/kargaranamir/hengam/blob/main/code/Train_HengamTransA.ipynb): This notebook uses the trained [HengamTransW.pth](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransW.pth) downloaded from HuggingFace and then trains it on [strong labeled data](https://huggingface.co/datasets/kargaranamir/HengamCorpus/raw/main/strong.txt) in an adverserial manner to produce [HengamTransA](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransA.pth).

## Data

### Train Data
HengamCorpus data along with strong labeled data is uploaded in [HuggingFace](https://huggingface.co/datasets/kargaranamir/HengamCorpus). Click on hyperlinks to download.
- HengamCorpus
  - [HengamCorpus train data](https://huggingface.co/datasets/kargaranamir/HengamCorpus/resolve/main/train.txt)
  - [HengamCorpus test data](https://huggingface.co/datasets/kargaranamir/HengamCorpus/resolve/main/test.txt)
  - [HengamCorpus validation data](https://huggingface.co/datasets/kargaranamir/HengamCorpus/resolve/main/val.txt)
- [Strong labeled data](https://huggingface.co/datasets/kargaranamir/HengamCorpus/raw/main/strong.txt)

### Evaluation Data
HengamGold, challenge set and prediction result of different models on HengamGold is available in [evaluation](./data/evaluation) folder.


## Citation

If you use any part of this repository in your research, please cite it using the following BibTex entry.
```python
@inproceedings{mirzababaei-etal-2022-hengam,
	title        = {Hengam: An Adversarially Trained Transformer for {P}ersian Temporal Tagging},
	author       = {Mirzababaei, Sajad  and Kargaran, Amir Hossein  and Sch{\"u}tze, Hinrich  and Asgari, Ehsaneddin},
	year         = 2022,
	booktitle    = {Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
	publisher    = {Association for Computational Linguistics},
	address      = {Online only},
	pages        = {1013--1024},
	url          = {https://aclanthology.org/2022.aacl-main.74}
}
```
