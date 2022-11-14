import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms.functional as F


class PRotLoader(datasets.FashionMNIST):
	def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
		super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

		self.args = args
		self.train = train
		
		if self.train:
			self.itransform = transforms.RandomCrop(self.args.pr_img_size, padding=self.args.padding)
			self.ptransform = transforms.RandomCrop(self.args.pr_img_crop_size)
			self.get_patch = transforms.RandomCrop(self.args.patch_size)

		else:
			self.itransform = transforms.CenterCrop(self.args.pr_img_size)
			self.ptransform = transforms.CenterCrop(self.args.pr_img_crop_size)
			self.get_patch = transforms.CenterCrop(self.args.patch_size)

		self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])
		self.to_tensor = transforms.ToTensor()

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img.numpy(), mode='L')

		prot_label = torch.full((5, self.args.n_pr_patches + 1), -100)

		# # Image
		prot_label[:4, 0] = torch.Tensor([0, 1, 2, 3])
		imgs = []
		for i in range(4):
			img_ = self.transform(img)
			img_ = self.itransform(img_)
			img_ = img_.rotate(i * 90)
			img_ = self.out_transform(img_)
			imgs.append(img_)

		# Patches
		img = self.transform(img)
		img = self.ptransform(img)

		p_angles = np.random.randint(low=0, high=4, size=self.args.n_pr_patches)

		prot_label[4, 1:] = torch.Tensor(p_angles)

		img_ = []
		idx = 0
		for j in range(0, self.args.n_pr_patches_):
			for i in range(0, self.args.n_pr_patches_):
				patch = img.crop((i * self.args.patch_crop_size, j * self.args.patch_crop_size, (i + 1) * self.args.patch_crop_size, (j + 1) * self.args.patch_crop_size))
				patch = self.get_patch(patch).rotate(p_angles[idx] * 90)
				patch = self.to_tensor(patch)
				img_.append(patch)
				idx = idx + 1
		img_ = vutils.make_grid(img_, nrow=self.args.n_pr_patches_, normalize=False, padding=0)
		img_ = F.normalize(img_, self.args.mean, self.args.std)[0].unsqueeze(0)
		imgs.append(img_)

		return torch.stack(imgs), prot_label
