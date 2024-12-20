# PatchRot
Official Implementation of paper [PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction](https://bmvc2024.org/proceedings/391/). <br>

## Table of Contents
- [Introduction](#introduction)
  - [Short Summary](#short-summary)
  - [Overview](#overview)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Run Commands](#run-commands)
  - [Data](#data)
- [Results](#results)
- [Cite](#cite)

## Introduction
### Short Summary
PatchRot is a self-supervised learning technique designed for Vision Transformers. It leverages image and patch rotation tasks to train networks to predict rotation angles, learning both global and patch-level representations.
<p align="center">
<img src="/figures/Toy.jpg" width="90%"></img>
</p>

### Overview
- PatchRot introduces a novel self-supervised strategy to learn rich and transferrable features.
- Rotates images and image patches by 0°, 90°, 180°, or 270°.
- Trains the network to predict the rotation angles of images and image patches as a classification task.
- Incorporates a buffer between patches to prevent trivial solutions such as edge continuity.
- Employs pretraining at smaller resolutions, followed by finetuning at the original size.
- This approach encourages the model to learn both global and patch-level representations.
- PatchRot was evaluated using the [DeiT-Tiny Transformer](https://arxiv.org/abs/2012.12877), with dataset-specific modifications to patch size.

## Usage
### Requirements
**Python** (>= 3.8),  **scikit-learn**,  **PyTorch** (>= 1.10), **torchvision**, and **timm** (for defining Vision Transformers, can be replaced with other frameworks)

### Run commands:
To pre-train and finetune models using PatchRot, run the following commands. Examples are provided for CIFAR10 (also available in <a href="run_cifar10.sh">run_cifar10.sh</a>).
- PatchRot Pretraining
```bash
python main_pretrain.py --dataset cifar10
```
- Finetuning Pretrained Model
```bash
python main_finetune.py --dataset cifar10 --init patchrot
```
- To train a baseline (Without PatchRot) set init to none: `python main_finetune.py --dataset cifar10 --init none`

We used a [DeiT-Tiny Transformer](https://arxiv.org/abs/2012.12877) and modified the patch size based on the dataset (refer <a href="https://github.com/s-chh/PatchRot/tree/main/config">config</a> folder).

### Data
- To change the dataset, **replace CIFAR10** with the appropriate dataset. <br>
- **CIFAR10**, **CIFAR100**, **FashionMNIST**, and **SVHN** are automatically downloaded by the script.
- **TinyImageNet**, **Animals10n**, and **Imagenet100** need to be downloaded manually (links below).
	- [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)  
	- [Animals10N](https://dm.kaist.ac.kr/datasets/animal-10n/)  
	- [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100)  
- For manually downloaded datasets, use the `--data_path` argument to specify the path to the dataset. Example:
```bash
python main_pretrain.py --dataset tinyimagenet --data_path /path/to/data
```

## Results
PatchRot significantly improves performance across diverse datasets. The table below compares the classification accuracy of baseline training (without PatchRot pretraining) and training with PatchRot pretraining:

| Dataset          | Without PatchRot Pretraining | With PatchRot Pretraining   |
|:-----------------|:----------------------------:|:---------------------------:|
| CIFAR10          | 84.4%                        | 91.3%                       |
| CIFAR100         | 56.5%                        | 66.7%                       |
| FashionMNIST     | 93.4%                        | 94.6%                       |
| SVHN             | 92.9%                        | 96.4%                       |
| Animals10N       | 69.6%                        | 79.5%                       |
| TinyImageNet     | 38.4%                        | 48.8%                       |
| ImageNet100      | 64.6%                        | 75.4%                       |


## Cite
If you found our work/code helpful, please cite our paper:
```
@inproceedings{Chhabra_2024_BMVC,
author    = {Sachin Chhabra and Hemanth Venkateswara and Baoxin Li},
title     = {PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction},
booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
publisher = {BMVA},
year      = {2024},
url       = {https://papers.bmvc2024.org/0391.pdf}
}
```
