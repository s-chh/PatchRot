import os
import json
import argparse
import datetime
from utils import print_args
from solver_pretrain import Solver


def main(args):
	os.makedirs(args.model_path, exist_ok=True)

	solver = Solver(args)
	solver.patchrot_train()


def update_args(args):
	with open(os.path.join("config", args.dataset+".json")) as data_file:
		config = json.load(data_file)

	args.patch_size = config["patch_size"]
	args.epochs = config["pretrain_epochs"]
	args.batch_size = config["batch_size"]
	args.lr = config["pretrain_lr"]
	args.warmup = config["pretrain_warmup"]
	args.weight_decay = config["weight_decay"]
	args.image_size = config["image_size"]

	args.hflip = config["hflip"]
	args.randomresizecrop = config["randomresizecrop"]
	args.padding = config["padding"]
	args.resizecrop = config["resizecrop"]
	args.n_channels = config["n_channels"]
	args.n_classes = config["n_classes"]
	args.cm = config["cm"]
	args.mean = config["mean"]
	args.std = config["std"]
	
	args.model_path = os.path.join(args.model_path, args.dataset, 'patchrot')
	args.data_path = os.path.join(args.data_path, args.dataset)

	args.buffer_size = args.patch_size

	args.n_patches_ = args.image_size // args.patch_size
	args.n_patches = args.n_patches_ ** 2
	
	args.patch_crop_size = args.patch_size + args.buffer_size

	args.n_patchrot_patches_ = args.image_size//args.patch_crop_size
	args.n_patchrot_patches = args.n_patchrot_patches_ ** 2
	
	args.input_size = args.n_patchrot_patches_ * args.patch_size

	return args


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_workers', type=int, default=4)
	
	parser.add_argument('--dataset', type=str.lower, default='cifar10')  # cifar10, cifar100, svhn, fashionmnist, refer config files
	parser.add_argument('--model_path', type=str, default='./saved_models/')	
	parser.add_argument('--data_path', type=str, default='./data')

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
