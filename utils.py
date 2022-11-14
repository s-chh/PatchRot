import math


def adjust_learning_rate(args, optimizer, epoch, max_epochs):
	lr = args.lr
	if epoch < args.warmup:
		lr = lr * epoch / args.warmup
	else:
		lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (max_epochs - args.warmup)))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

