# PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction
This is the official PyTorch Implementation of our upcoming BMVC 2024 PatchRot paper "PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction". <br>

## Introduction
PatchRot rotates images and image patches and trains the network to predict the rotation angles. 
The network learns to extract global image and patch-level features through this process. 
PatchRot pretraining extracts superior features and provides improved performance. <br>

## Run commands (also available in <a href="run_cifar10.sh">run_cifar10.sh</a>):
Run <strong>main_pretrain.py</strong> to pre-train the network with PatchRot, followed by <strong>main_finetune.py --init patchrot</strong> to finetune the network.<br>
<strong>main_finetune.py --init none</strong> can be used to train the network without any pretraining (training from random initialization).<br>
Below is an example on CIFAR10:

| Method | Run Command |
| :---         | :---         |
| PatchRot pretraining | python main_pretrain.py --dataset cifar10 |
| Finetuning pretrained model | python main_finetune.py --dataset cifar10 --init patchrot |
| Training from random init | python main_finetune.py --dataset cifar10 --init none |

Replace cifar10 with the appropriate dataset. <br>
Supported datasets: CIFAR10, CIFAR100, FashionMNIST, SVHN, TinyImageNet, Animals10N, and ImageNet100. <br><br>
CIFAR10, CIFAR100, FashionMNIST, and SVHN datasets will be downloaded to the path specified in the "data_path" argument (default: "./data").<br>
TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument. 

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
