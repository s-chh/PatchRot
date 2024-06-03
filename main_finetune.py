import os
import math
import json
import datetime
import argparse
from utils import print_args
from solver_finetune import Solver


def main(args):
	os.makedirs(args.model_path, exist_ok=True)

	solver = Solver(args)

	solver.train()
	solver.test(train=True)


def update_args(args):
	with open(os.path.join("config", args.dataset+".json")) as data_file:
		config = json.load(data_file)

	if args.init == 'patchrot':
		args.epochs = config["finetune_epochs"]
	else:
		args.epochs = config["scratch_epochs"]

	args.patch_size = config["patch_size"]
	args.batch_size = config["batch_size"]
	args.lr = config["finetune_lr"]
	args.warmup = config["finetune_warmup"]
	args.image_size = config["image_size"]

	args.weight_decay = config["weight_decay"]
	args.hflip = config["hflip"]
	args.randomresizecrop = config["randomresizecrop"]
	args.padding = config["padding"]
	args.resizecrop = config["resizecrop"]
	args.n_channels = config["n_channels"]
	args.n_classes = config["n_classes"]
	args.cm = config["cm"]
	args.mean = config["mean"]
	args.std = config["std"]
	
	args.data_path = os.path.join(args.data_path, args.dataset)

	if args.init == 'patchrot':
		args.model_path = os.path.join(args.model_path, args.dataset, 'patchrot')	
		args.model_name = args.init + "_" + 'sup_' + str(args.n_layers_to_freeze) + '.pt'

		args.buffer_size = args.patch_size
		args.patch_crop_size = math.ceil(args.patch_size + args.buffer_size)
		args.n_patchrot_patches_ = args.image_size//args.patch_crop_size
		args.model_input_size = args.n_patchrot_patches_ * args.patch_size

	else:
		args.model_path = os.path.join(args.model_path, args.dataset, 'sup')
		args.model_name = 'sup.pt'
	
		args.model_input_size = args.image_size

	return args


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_workers', type=int, default=4)
	
	parser.add_argument('--dataset', type=str.lower, default='cifar10')
	parser.add_argument('--model_path', type=str, default='./saved_models/')
	parser.add_argument('--data_path', type=str, default='./data')

	parser.add_argument('--init', type=str, default='patchrot')  # | 'none' refers to training from scratch | 'patchrot'
	parser.add_argument('--n_layers_to_freeze', type=int, default=0)

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
