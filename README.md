# PatchRot
Official PyTorch Implementation of our upcoming BMVC 2024 PatchRot paper "PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction". <br>

## Introduction
- Self-supervised training strategy for Vision Transformers to learn rich and transferrable features.
- PatchRot rotates images and image patches by 0°, 90°, 180°, or 270°.
- Trains the network to predict the rotation angles of images and image patches via classification.
- Use a buffer between the patches to avoid trivial solutions like edge continuity.
- Pre-train at a smaller size and followed by finetuning at the original size.
- Learn global and patch-level information of images.

## Run commands (also available in <a href="run_cifar10.sh">run_cifar10.sh</a>):
- Run  <strong>```main_pretrain.py```</strong> to pre-train the network with PatchRot.
- Next <strong>```main_finetune.py --init patchrot```</strong> to finetune the network.
Below is an example on CIFAR10:

| Method | Run Command |
| :---         | :---         |
| PatchRot pretraining | python main_pretrain.py --dataset cifar10 |
| Finetuning pretrained model | python main_finetune.py --dataset cifar10 --init patchrot |

Replace cifar10 with the appropriate dataset. <br>
Supported datasets: CIFAR10, CIFAR100, FashionMNIST, SVHN, TinyImageNet, Animals10N, and ImageNet100. <br><br>
CIFAR10, CIFAR100, FashionMNIST, and SVHN datasets will be downloaded to the path specified in the "data_path" argument (default: "./data").<br>
TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument. 

- <strong>main_finetune.py --init none</strong> can be used to train the network without any pretraining (training from random initialization).<br>
| Training from random init | python main_finetune.py --dataset cifar10 --init none |


## Results
| Dataset | Without PatchRot Pretraining | With PatchRot Pretraining |
| :---         |     :---:      |     :---:      |
| CIFAR10 | 84.4 | 91.3 |
| CIFAR100 | 56.5 | 66.7 |
| FashionMNIST | 93.4 | 94.6|
| SVHN | 92.9 | 96.4 |
| Animals10N | 69.6 | 79.5 |
| TinyImageNet | 38.4 | 48.8 |
| ImageNet100 | 64.6 | 75.4 |
