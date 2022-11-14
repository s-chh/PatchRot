import torch
from torchvision import transforms
import os
import torch.utils.data as data
from prot_loaders import cifar10, cifar100, fmnist, im, svhn, mnist
from torchvision import datasets


def prot_loader(args):
	train_transform = []
	train_transform.append(transforms.Resize([args.image_size, args.image_size]))
	if args.hflip:
		train_transform.append(transforms.RandomHorizontalFlip())
	if args.num_channels == 1:
		train_transform.append(transforms.Grayscale(1))
	train_transform = transforms.Compose(train_transform)

	test_transform = []
	test_transform.append(transforms.Resize([args.image_size, args.image_size]))
	if args.num_channels == 1:
		test_transform.append(transforms.Grayscale(1))
	test_transform = transforms.Compose(test_transform)

	if args.ds == 'cifar10':
		train = cifar10.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=True, transform=train_transform, download=True)
		test = cifar10.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=False, transform=test_transform, download=True)
	elif args.ds == 'cifar100':
		train = cifar100.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=True, transform=train_transform, download=True)
		test = cifar100.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=False, transform=test_transform, download=True)
	elif args.ds == 'fmnist':
		train = fmnist.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=True, transform=train_transform, download=True)
		test = fmnist.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=False, transform=test_transform, download=True)
	elif args.ds == 'svhn':
		train = svhn.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), split='train', transform=train_transform, download=True)
		test = svhn.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), split='test', transform=test_transform, download=True)
	elif args.ds == 'mnist':
		train = mnist.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=True, transform=train_transform, download=True)
		test = mnist.PRotLoader(args, root=os.path.join(args.ds_path, args.ds), train=False, transform=test_transform, download=True)
	else:
		train = im.PRotLoader(args, root=os.path.join(args.ds_path, args.ds, 'train'), transform=train_transform, train=True)
		test = im.PRotLoader(args, root=os.path.join(args.ds_path, args.ds, 'train'), transform=test_transform, train=False)

	train_loader = torch.utils.data.DataLoader(dataset=train,
												batch_size=args.batch_size,
												shuffle=True,
												num_workers=args.num_workers,
												drop_last=True)

	test_loader = torch.utils.data.DataLoader(dataset=test,
												batch_size=args.batch_size*3,
												shuffle=True,
												num_workers=args.num_workers,
												drop_last=False)

	return train_loader, test_loader


def cls_loader(args):
	train_transform = []
	train_transform.append(transforms.Resize([args.image_size, args.image_size]))
	train_transform.append(transforms.RandomCrop(args.image_size, padding=args.padding))
	if args.hflip:
		train_transform.append(transforms.RandomHorizontalFlip())
	if args.num_channels == 1:
		train_transform.append(transforms.Grayscale(1))
	train_transform.append(transforms.ToTensor())
	train_transform.append(transforms.Normalize(args.mean, args.std))
	train_transform = transforms.Compose(train_transform)

	test_transform = []
	test_transform.append(transforms.Resize([args.image_size, args.image_size]))
	if args.num_channels == 1:
		test_transform.append(transforms.Grayscale(1))
	test_transform.append(transforms.ToTensor())
	test_transform.append(transforms.Normalize(args.mean, args.std))
	test_transform = transforms.Compose(test_transform)

	if args.ds == 'cifar10':
		train = datasets.CIFAR10(root=args.ds_path, train=True, transform=train_transform, download=True)
		test = datasets.CIFAR10(root=args.ds_path, train=False, transform=test_transform, download=True)
	elif args.ds == 'cifar100':
		train = datasets.CIFAR100(root=args.ds_path, train=True, transform=train_transform, download=True)
		test = datasets.CIFAR100(root=args.ds_path, train=False, transform=test_transform, download=True)
	elif args.ds == 'fmnist':
		train = datasets.FashionMNIST(root=args.ds_path, train=True, transform=train_transform, download=True)
		test = datasets.FashionMNIST(root=args.ds_path, train=False, transform=test_transform, download=True)
	elif args.ds == 'svhn':
		train = datasets.SVHN(root=args.ds_path, split='train', transform=train_transform, download=True)
		test = datasets.SVHN(root=args.ds_path, split='test', transform=test_transform, download=True)
	elif args.ds == 'mnist':
		train = datasets.MNIST(root=args.ds_path, train=True, transform=train_transform, download=True)
		test = datasets.MNIST(root=args.ds_path, train=False, transform=test_transform, download=True)
	else:
		train = datasets.ImageFolder(root=os.path.join(args.ds_path, 'train'), transform=train_transform)
		test = datasets.ImageFolder(root=os.path.join(args.ds_path, 'val'), transform=test_transform)

	train_loader = torch.utils.data.DataLoader(dataset=train,
												batch_size=args.batch_size,
												shuffle=True,
												num_workers= args.num_workers,
												drop_last=True)

	test_loader = torch.utils.data.DataLoader(dataset=test,
												batch_size=args.batch_size*3,
												shuffle=True,
												num_workers=args.num_workers,
												drop_last=False)

	return train_loader, test_loader

