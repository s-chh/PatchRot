# PatchRot

## Introduction
This is the official PyTorch Implementation of our PatchRot paper [PatchRot: Self-Supervised Training of Vision Transformers by Rotation Prediction](https://arxiv.org/abs/2210.15722)
PatchRot rotates images and image patches and trains the network to predict the rotation angles. 
Through this process, the network learns to extract both global image and patch-level features. 
PatchRot pretraining extracts superior features and results in improved performance. <br>

## Run commands:
Run <strong>main_pretrain.py</strong> to pre-train the network with PatchRot, followed by <strong>main_finetune.py --init patchrot</strong> to finetune the network.<br>
<strong>main_finetune.py --init none</strong> can be used to train the network without any pretraining (training from random initialization).<br>
Below is an example on CIFAR10:
<table>
  <tr>
    <th>Method</th>
    <th>Run Command</th>
  </tr>
  <tr>
    <td>PatchRot Pretraining</td>
    <td>python -u main_pretrain.py --dataset cifar10</td>
  </tr>
  <tr>
    <td>Finetuning Pretrained Model</td>
    <td>python -u main_finetune.py --dataset cifar10 --init patchrot</td>
  </tr>
  <tr>
    <td>Training from random init</td>
    <td>python -u main_finetune.py --dataset cifar10 --init none</td>
  </tr>
</table>
Replace cifar10 with the appropriate dataset. <br>
Supported datasets: CIFAR10, CIFAR100, FashionMNIST, SVHN, TinyImageNet, Animals10N, and ImageNet100. <br>
CIFAR10, CIFAR100, FashionMNIST, and SVHN datasets will be downloaded to the path specified in the "data_path" argument (default: "./data").<br>
TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument. 

## Results
<table>
  <tr>
    <th>Dataset</th>
    <th>Without Pretraining</th>
    <th>With Pretraining</th>
  </tr>
  <tr>
    <td>CIFAR10</td>
    <td>84.4</td>
    <td>91.3</td>
  </tr>
  <tr>
    <td>CIFAR100</td>
    <td>56.5</td>
    <td>66.7</td>
  </tr>
  <tr>
    <td>FashionMNIST</td>
    <td>93.4</td>
    <td>94.6</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>92.9</td>
    <td>96.4</td>
  </tr>
  <tr>
    <td>Animals10N</td>
    <td>69.6</td>
    <td>79.5</td>
  </tr>
  <tr>
    <td>TinyImageNet</td>
    <td>38.4</td>
    <td>48.8</td>
  </tr>
  <tr>
    <td>ImageNet100</td>
    <td>64.6</td>
    <td>75.4</td>
  </tr>
</table>

## Citation
If you use this method or this code in your paper, then please cite it:

```
@article{chhabra2022patchrot,
  title={PatchRot: A Self-Supervised Technique for Training Vision Transformers},
  author={Chhabra, Sachin and Dutta, Prabal Bijoy and Venkateswara, Hemanth and Li, Baoxin},
  journal={arXiv preprint arXiv:2210.15722},
  year={2022}
}
```
