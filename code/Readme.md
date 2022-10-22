# Hengam Codes

## HengamTagger
The HengamTagger is distributed as [Parstdex](https://github.com/kargaranamir/parstdex) package (will be continuously updated) and is available via [pip](https://pypi.org/project/parstdex).

## HengamTransformer
In this directory 4 different notebooks are provided, google colab compatibilty is checked.
- Inference_HengamTransW.ipynb: In this notebook, [HengamTransW.pth](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransW.pth) is downloaded from [Hengam HuggingFace model card](https://huggingface.co/kargaranamir/Hengam), and an inference is provided in the last cells.
- Inference_HengamTransA.ipynb: In this notebook, [HengamTransA.pth](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransA.pth) is downloaded from [Hengam HuggingFace model card](https://huggingface.co/kargaranamir/Hengam), and an inference is provided in the last cells.
- Train_HengamTransW.ipynb: This notebook uses the [HengamCorpus](https://huggingface.co/datasets/kargaranamir/HengamCorpus) dataset uploaded on HuggingFace. Afterwards, the HengamTransW will be trained.
- Train_HengamTransA.ipynb: This notebook uses the trained [HengamTransW.pth](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransW.pth) downloaded from HuggingFace and then trains it on [strong labeled data](https://huggingface.co/datasets/kargaranamir/HengamCorpus/raw/main/strong.txt) in an adverserial manner to produce [HengamTransA](https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransA.pth).
