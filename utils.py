import torch.nn as nn
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


def stop_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def allow_grad(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True


def reshape_batch(data):
    x, y = zip(*data)
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y


class PatchRot(nn.Module):
    def __init__(self, dataset, transform, out_transform, image_size, patch_size, buffer_size, train=True):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.image_size = image_size
        self.patch_size = patch_size
        self.buffer_size = buffer_size
        self.train = train

        self.patch_crop_size = self.patch_size + self.buffer_size
        self.n_patchrot_patches_ = self.image_size // self.patch_crop_size
        self.n_patchrot_patches = self.n_patchrot_patches_ ** 2
        self.input_size = self.n_patchrot_patches_ * self.patch_size
        self.patchrot_img_crop_size = (self.image_size // self.patch_crop_size) * self.patch_crop_size

        self.resize_transform = transforms.Resize(self.image_size)

        if self.train:
            self.img_transform = transforms.RandomCrop(self.input_size)
            self.patchrot_img_transform = transforms.RandomCrop(self.patchrot_img_crop_size)
            self.get_patch = transforms.RandomCrop(self.patch_size)

        else:
            self.img_transform = transforms.CenterCrop(self.input_size)
            self.patchrot_img_transform = transforms.CenterCrop(self.patchrot_img_crop_size)
            self.get_patch = transforms.CenterCrop(self.patch_size)

        self.out_transform = out_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)

        patchrot_label = torch.full((5, self.n_patchrot_patches + 1), -100)

        # Image-level
        imgs = []
        patchrot_label[:4, 0] = torch.Tensor([0, 1, 2, 3])
        for i in range(4):
            img_ = self.transform(img)
            img_ = self.img_transform(img_)
            img_ = img_.rotate(i * 90)
            img_ = self.out_transform(img_)
            imgs.append(img_)

        # Patch-level
        img = self.resize_transform(img)
        img = self.patchrot_img_transform(img)

        p_angles = np.random.randint(low=0, high=4, size=self.n_patchrot_patches)

        patchrot_label[-1, 1:] = torch.Tensor(p_angles)

        img_ = Image.new(img.mode, (self.input_size, self.input_size))
        idx = 0
        for j in range(self.n_patchrot_patches_):
            for i in range(self.n_patchrot_patches_):
                patch = img.crop((i * self.patch_crop_size, j * self.patch_crop_size, (i + 1) * self.patch_crop_size,
                                  (j + 1) * self.patch_crop_size))
                patch = self.get_patch(patch).rotate(p_angles[idx] * 90)
                x_loc = (idx // self.n_patchrot_patches_) * self.patch_size
                y_loc = (idx % self.n_patchrot_patches_) * self.patch_size
                img_.paste(patch, (y_loc, x_loc))
                idx = idx + 1

        img_ = self.out_transform(img_)
        imgs.append(img_)

        return torch.stack(imgs), patchrot_label
