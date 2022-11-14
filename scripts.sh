python -u main.py --ds fmnist --method full --num_layers_to_freeze 0

## Or ##

python -u main.py --ds cifar10 --method patchrot
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 0
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 1
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 2
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 3
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 4
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 5
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 6
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 7
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 8
python -u main.py --ds cifar10 --method finetune --num_layers_to_freeze 9

