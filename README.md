# PatchRot
Official Implementation of paper "PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction". <br>

## Introduction
Summary: Self-supervised technique for vision transformers that predicts rotation angles of images and patches.
<p align="center">
<img src="/figures/Toy.jpg" width="90%"></img>
</p>

**Overview**
- Self-supervised training strategy for vision transformers to learn rich and transferrable features.
- PatchRot rotates images and image patches by 0°, 90°, 180°, or 270°.
- Trains the network to predict the rotation angles of images and image patches via classification.
- Use a buffer between the patches to avoid trivial solutions like edge continuity.
- Pre-train at a smaller size, followed by finetuning at the original size.
- Learn global and patch-level information of images.

## Run commands (also available in <a href="run_cifar10.sh">run_cifar10.sh</a>):
- Run <strong>`main_pretrain.py`</strong> to pre-train the network with PatchRot.
- Next <strong>`main_finetune.py --init patchrot`</strong> to finetune the network.
Below is an example on CIFAR10:

| Method | Run Command |
| :---         | :---         |
| PatchRot pretraining | python main_pretrain.py --dataset cifar10 |
| Finetuning pretrained model | python main_finetune.py --dataset cifar10 --init patchrot |
- For baseline training (random init) use <strong>`main_finetune.py --dataset cifar10 --init none`</strong>
- We used a **DeiT-Tiny transformer** for the experiments and modified the patch size based on the dataset.
   - Details are available in <a href="https://github.com/s-chh/PatchRot/tree/main/config">config</a>
- To change the dataset, **replace cifar10** with the **appropriate dataset**. <br>
   - Cifar10, Cifar100, FashionMNIST, and SVHN will be auto-downloaded.
   - TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument.  

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
