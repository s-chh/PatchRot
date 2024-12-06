# PatchRot
Official Implementation of paper "PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction". <br>

## Table of Contents
- [Introduction](#introduction)
- [Overview](#overview)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Run Commands](#run-commands)
  - [Data](#data)
- [Results](#results)
- [Cite](#cite)

## Introduction
**Short summary**: PatchRot is a self-supervised learning technique designed for Vision Transformers. It leverages image and patch rotation tasks to train networks to predict rotation angles, learning both global and patch-level representations.
<p align="center">
<img src="/figures/Toy.jpg" width="90%"></img>
</p>

### Overview
- Self-supervised introduces a novel self-supervised strategy to learn rich and transferrable features.
- PatchRot rotates images and image patches by 0°, 90°, 180°, or 270°.
- Trains the network to predict the rotation angles of images and image patches as a classification task.
- Incorporates a buffer between patches to prevent trivial solutions such as edge continuity.
- Employs pretraining at smaller resolutions, followed by finetuning at the original size.
- This approach encourages the model to learn both global and local representations, improving its performance on downstream tasks.
- PatchRot was evaluated using the DeiT-Tiny Transformer architecture, with dataset-specific modifications to patch size.

## Usage
### Requirements
- Python
- scikit-learn
- PyTorch
- torchvision
- timm for defining Vision Transformer (can be replaced with other network definition)

### Run commands (also available in <a href="run_cifar10.sh">run_cifar10.sh</a>):
- Run <strong>`main_pretrain.py`</strong> to pre-train the network with PatchRot.
- Next <strong>`main_finetune.py --init patchrot`</strong> to finetune the network.
Below is an example on CIFAR10:

| Method | Run Command |
| :---         | :---         |
| PatchRot pretraining | python main_pretrain.py --dataset cifar10 |
| Finetuning pretrained model | python main_finetune.py --dataset cifar10 --init patchrot |
- For baseline training (random init) use <strong>`main_finetune.py --dataset cifar10 --init none`</strong>
- We used a **DeiT-Tiny transformer** for the experiments and modified the patch size based on the dataset.
   - Details are available in <a href="https://github.com/s-chh/PatchRot/tree/main/config">config</a> folder.

### Data
- To change the dataset, **replace cifar10** with the appropriate dataset. <br>
- **Cifar10**, **Cifar100**, **FashionMNIST**, and **SVHN** will be auto-downloaded.
- **TinyImageNet**, **Animals10n**, and **Imagenet100** need to be downloaded.
   - Data must be split into 'train' and 'test' folders. 
   - Path needs to be provided using the "data_path" argument.
- Dataset links:
   - TinyImageNet: <a href="http://cs231n.stanford.edu/tiny-imagenet-200.zip">http://cs231n.stanford.edu/tiny-imagenet-200.zip</a> 
   - Animals10N: <a href="https://dm.kaist.ac.kr/datasets/animal-10n/">https://dm.kaist.ac.kr/datasets/animal-10n/</a>  
   - ImageNet100: <a href="https://www.kaggle.com/datasets/ambityga/imagenet100">https://www.kaggle.com/datasets/ambityga/imagenet100/</a>  

## Results
| Dataset | Without PatchRot Pretraining | With PatchRot Pretraining |
| :---         |     :---:      |     :---:      |
| Cifar10 | 84.4 | 91.3 |
| Cifar100 | 56.5 | 66.7 |
| FashionMNIST | 93.4 | 94.6|
| SVHN | 92.9 | 96.4 |
| Animals10N | 69.6 | 79.5 |
| TinyImageNet | 38.4 | 48.8 |
| ImageNet100 | 64.6 | 75.4 |

## Cite
If you found our work/code helpful, please cite our paper:
```
Bibtex upcoming
```
