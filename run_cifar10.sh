python -u main_pretrain.py --dataset cifar10 &> pretrain.txt
python -u main_finetune.py --dataset cifar10 --init patchrot &> finetune.txt
python -u main_finetune.py --dataset cifar10 --init none &> scratch.txt
