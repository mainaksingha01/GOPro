# GOPro: Generate and Optimize Prompts in CLIP using Self-Supervised Learning

Official repository of GOPro.

## **British Machine Vision Conference (BMVC) 2023**

[![paper](https://img.shields.io/badge/Conference-Paper-blue)](https://papers.bmvc2023.org/0314.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-F9D371)](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0314_supp.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2308.11605.pdf)

## Abstract

Large-scale foundation models, such as CLIP, have demonstrated remarkable success in visual recognition tasks by embedding images in a semantically rich space. Selfsupervised learning (SSL) has also shown promise in improving visual recognition by learning invariant features. However, the combination of CLIP with SSL is found to face challenges due to the multi-task framework that blends CLIP’s contrastive loss and SSL’s loss, including difficulties with loss weighting and inconsistency among different views of images in CLIP’s output space. To overcome these challenges, we propose a prompt learning-based model called GOPro, which is a unified framework that ensures similarity between various augmented views of input images in a shared image-text embedding space, using a pair of learnable image and text projectors atop CLIP, to promote invariance and generalizability. To automatically learn such prompts, we leverage the visual content and style primitives extracted from pre-trained CLIP and adapt them to the target task. In addition to CLIP’s cross-domain contrastive loss, we introduce a visual contrastive loss and a novel prompt consistency loss, considering the different views of the images. GOPro is trained end-to-end on all three loss objectives, combining the strengths of CLIP and SSL in a principled manner. Empirical evaluations demonstrate that GOPro outperforms the state-of-the-art prompting techniques on three challenging domain generalization tasks across multiple benchmarks by a significant margin.

## Architecture

<img src="https://github.com/mainaksingha01/GOPro/blob/master/images/architecture.png" width="1000">

## Datasets
 Follow the instructions given in [CoOp.datasets](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the datasets and set up the dataloaders.
 
## Code

 - `datasets` folder contains the dataloader files of each datasets.
 - `trainers` folder contains the code of our model.
 - Clone the awesome toolbox of [dassl](https://github.com/KaiyangZhou/Dassl.pytorch/tree/master/dassl) inside this repo.
 - `scripts` folder holds the scripts of for training and testing.
 - Define the dataset and task (base2new, cross-dataset, domain-generalization) in the script command 

```shell (for example)
$ cd scripts
$ bash train.sh caltech101 basenew
$ bash test.sh caltech101 basenew
```

## Results

### Base-to-New Class Generalization

<img src="https://github.com/mainaksingha01/GOPro/blob/master/images/b2n.png" width="600">

### Cross Dataset Generalization

<img src="https://github.com/mainaksingha01/GOPro/blob/master/images/cd.png" width="600">

### Domain Generalization

<img src="https://github.com/mainaksingha01/GOPro/blob/master/images/dg.png" width="500">

## Bibtex

Please cite the paper if you use our work . Thanks.

```
@article{singha2023gopro,
  title={GOPRO: Generate and Optimize Prompts in CLIP using Self-Supervised Learning},
  author={Singha, Mainak and Jha, Ankit and Banerjee, Biplab},
  journal={arXiv preprint arXiv:2308.11605},
  year={2023}
}

```

## Acknowledgements

Thanks to the authors of [CoOp](https://github.com/KaiyangZhou/CoOp) as our code is mainly based on this repository.
