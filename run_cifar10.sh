python -u main_pretrain.py --dataset cifar10
python -u main_finetune.py --dataset cifar10 --init patchrot
python -u main_finetune.py --dataset cifar10 --init none
