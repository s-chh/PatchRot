import argparse
import os
from solver import Solver
import datetime
import math
import json


def main(args):
    os.makedirs(args.model_path, exist_ok=True)

    solver = Solver(args)

    if args.method == 'full' or args.method == 'patchrot':
        solver.prot_train()
        solver.prot_test(train=True)
    if args.method == 'full' or args.method == 'finetune':
        solver.cls_finetune()
        solver.cls_test(train=True)


def update_args(args):
    with open(os.path.join("config", args.ds + ".json")) as data_file:
        config = json.load(data_file)

    args.image_size = config["image_size"]
    args.hflip = config["hflip"]
    args.num_channels = config["num_channels"]
    args.num_classes = config["num_classes"]
    args.cm = config["cm"]
    args.padding = config["padding"]
    args.mean = config["mean"]
    args.std = config["std"]

    args.model_path = os.path.join(args.model_path, args.ds)
    args.ds_path = os.path.join(args.ds_path, args.ds)

    args.buffer_size = args.patch_size // 4
    args.patch_crop_size = math.ceil(args.patch_size + args.buffer_size)

    args.n_pr_patches_ = args.image_size // args.patch_crop_size
    args.n_pr_patches = args.n_pr_patches_ ** 2

    args.pr_img_crop_size = args.n_pr_patches_ * args.patch_crop_size
    args.pr_img_size = args.n_pr_patches_ * args.patch_size

    args.num_workers = os.cpu_count() - 2

    return args


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='full', choices=['full', 'patchrot', 'finetune'])

    parser.add_argument('--ds', type=str, default='cifar10')
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--num_layers_to_freeze', type=int, default=0)

    parser.add_argument('--pr_epochs', type=int, default=300)
    parser.add_argument('--ft_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=3e-2)

    parser.add_argument('--model_path', type=str, default='./model/')
    parser.add_argument('--ds_path', type=str, default='./data/')

    args = parser.parse_args()
    args = update_args(args)

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    print_args(args)
    main(args)

    end_time = datetime.datetime.now()
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    duration = end_time - start_time
    print("Duration: " + str(duration))
