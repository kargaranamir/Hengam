# Hengam: An Adversarially Trained Transformer for Persian Temporal Tagging

## HuggingFace
- [HengamTrans Space](https://huggingface.co/spaces/kargaranamir/Hengam)
- [HengamTagger (Parstdex) Space](https://huggingface.co/spaces/kargaranamir/parstdex)
- [HengamTrans Models (ModelCard)](https://huggingface.co/kargaranamir/Hengam)
- [HengamCorpus (Dataset)](https://huggingface.co/datasets/kargaranamir/HengamCorpus)


## Code (Software)

### HengamTagger
The HengamTagger is distributed as [Parstdex](https://github.com/kargaranamir/parstdex) package (will be continuously updated) and is available via [pip](https://pypi.org/project/parstdex). 
<p align="center">
  <a href="https://github.com/kargaranamir/parstdex/">
    <img src="https://user-images.githubusercontent.com/26163093/201153110-d14f95fe-020b-4557-984e-ff6957d86a41.png" width="400">
  </a>
</p>

### HengamTransformer
In this directory 4 different notebooks are provided. Click on the hyperlinks to open each in Google Colab.

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
    title = "Hengam: An Adversarially Trained Transformer for {P}ersian Temporal Tagging",
    author = {Mirzababaei, Sajad  and
      Kargaran, Amir Hossein  and
      Sch{\"u}tze, Hinrich  and
      Asgari, Ehsaneddin},
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing",
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.74",
    pages = "1013--1024",
}
```
