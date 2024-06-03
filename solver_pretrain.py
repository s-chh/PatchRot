import torch
import torch.nn as nn
import os
from torch import optim
from data_loader import patchrot_loader
from model import tiny_deit, patchrot_classifier


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = patchrot_loader(self.args)

        self.encoder = tiny_deit(self.args.input_size, self.args.patch_size, self.args.n_channels, self.args.n_classes, dynamic_img_size=True).cuda()
        print("Encoder:")
        print(self.encoder)

        self.patchrot_classifier = patchrot_classifier().cuda()
        print("\nPatchrot Classifier:")
        print(self.patchrot_classifier)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def patchrot_train(self):
        iter_per_epoch = len(self.train_loader)

        print(f"Iters per epoch: {iter_per_epoch}")

        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.patchrot_classifier.parameters()), self.args.lr,
                                betas=(0.9, 0.95), weight_decay=self.args.weight_decay)

        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / self.args.warmup, end_factor=1.0, total_iters=self.args.warmup - 1, last_epoch=-1, verbose=False)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs - self.args.warmup, eta_min=1e-5, verbose=False)

        for epoch in range(self.args.epochs):
            if epoch < self.args.warmup:
                if epoch > 0:
                    linear_warmup.step()
                print(f"\nEp:[{epoch + 1}/{self.args.epochs}]\tlr: {linear_warmup.get_last_lr()[0]:.6f}")
            else:
                cos_decay.step()
                print(f"\nEp:[{epoch + 1}/{self.args.epochs}]\tlr: {cos_decay.get_last_lr()[0]:.6f}")

            self.encoder.train()
            self.patchrot_classifier.train()

            for i, data in enumerate(self.train_loader):
                x, y_patchrot = data
                x, y_patchrot = x.cuda(), y_patchrot.cuda()

                features = self.encoder.forward_features(x)
                logits = self.patchrot_classifier(features)

                imgs_rows = y_patchrot[:, 0] != -100
                patch_rows = y_patchrot[:, 0] == -100

                img_logits = logits[imgs_rows, 0].reshape(-1, 4)
                img_targets = y_patchrot[imgs_rows, 0].reshape(-1)
                imgs_loss = self.ce(img_logits, img_targets)

                patch_logits = logits[patch_rows, 1:].reshape(-1, 4)
                patch_targets = y_patchrot[patch_rows, 1:].reshape(-1)
                patch_loss = self.ce(patch_logits, patch_targets)
                loss = (imgs_loss + patch_loss) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    print(f"It: {i + 1}/{iter_per_epoch}, i_loss:{imgs_loss:.4f}, p_loss:{patch_loss:.4f}, t_loss:{loss:.4f}")

            if (epoch + 1) % 25 == 0:
                torch.save(self.encoder.state_dict(), os.path.join(self.args.model_path, 'patchrot_encoder.pt'))
                torch.save(self.patchrot_classifier.state_dict(), os.path.join(self.args.model_path, 'patchrot_classifier.pt'))
