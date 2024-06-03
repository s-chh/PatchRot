import torch
import torch.nn as nn
import os
from torch import optim
from sklearn.metrics import confusion_matrix, accuracy_score
from utils import stop_grad, allow_grad
from data_loader import sup_loader
from model import tiny_deit


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = sup_loader(self.args)

        self.net = tiny_deit(self.args.model_input_size, self.args.patch_size, self.args.n_channels, self.args.n_classes, dynamic_img_size=True).cuda()

        if self.args.init == 'patchrot':
            self.net.load_state_dict(torch.load(os.path.join(self.args.model_path, 'patchrot_encoder.pt')))

        print("Encoder:")
        print(self.net)

        self.ce = nn.CrossEntropyLoss()
        self.set_grad()

    def set_grad(self):
        count = 0

        if self.args.n_layers_to_freeze == 0:
            allow_grad(self.net)
            print("Training whole network.")
        else:
            stop_grad(self.net)
            count += 1
            print(f"Patch Embed block will not be trained.")

            for i in range(12):
                if count < self.args.n_layers_to_freeze:
                    stop_grad(self.net.blocks[i])
                    count += 1
                    print(f"Encoder Block {i + 1}/12 will not be trained.")
                else:
                    allow_grad(self.net.blocks[i])
                    print(f"Encoder Block {i + 1}/12 will be trained.")

            allow_grad(self.net.head)
            allow_grad(self.net.head_drop)
            allow_grad(self.net.fc_norm)
            allow_grad(self.net.norm)

    def train_mode(self):
        count = 0

        if self.args.n_layers_to_freeze == 0:
            self.net.train()
            return
        else:
            self.net.eval()

            for i in range(12):
                if count < self.args.n_layers_to_freeze:
                    self.net.blocks[i].eval()
                    count += 1
                else:
                    self.net.blocks[i].train()

            self.net.head.train()
            self.net.head_drop.train()
            self.net.fc_norm.train()
            self.net.norm.train()

    def train(self):
        iter_per_epoch = len(self.train_loader)

        print(f"Iters per epoch: {iter_per_epoch:d}")

        if self.args.init == 'patchrot':
            lr_coefs = [0.75 ** (12 + 1 - i) for i in range(12 + 1)]

            params_to_train = []
            for name, p in self.net.named_parameters():

                if not p.requires_grad:
                    continue

                if name in ['cls_token', 'pos_embed', 'pos_drop', 'patch_drop', 'norm_pre'] or 'patch_embed' in name:
                    coef_to_use = lr_coefs[0]
                    lr = self.args.lr * coef_to_use
                    params_to_train.append({'params': p, lr: lr})
                    print(f"Coef of learning rate for {name} of embedding block: {coef_to_use:.4f}")

                elif 'blocks' in name:
                    block_index = int(name.split(".")[1])
                    coef_to_use = lr_coefs[block_index + 1]
                    lr = self.args.lr * coef_to_use
                    params_to_train.append({'params': p, lr: lr})
                    print(f"Coef of learning rate for {name} of encoder block {block_index + 1}: {coef_to_use:.4f}")

                else:
                    lr = self.args.lr
                    params_to_train.append({'params': p, lr: lr})
                    print(f"Coef of learning rate for {name}: 1")
        else:
            params_to_train = self.net.parameters()

        optimizer = optim.AdamW(params_to_train, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / self.args.warmup, end_factor=1.0, total_iters=self.args.warmup - 1, last_epoch=-1, verbose=False)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs - self.args.warmup, eta_min=1e-5, verbose=False)

        best_acc = 0
        for epoch in range(self.args.epochs):
            if epoch < self.args.warmup:
                if epoch > 0:
                    linear_warmup.step()
                print(f"\nEp:[{epoch + 1}/{self.args.epochs}]\tlr: {linear_warmup.get_last_lr()[0]:.6f}")
            else:
                cos_decay.step()
                print(f"\nEp:[{epoch + 1}/{self.args.epochs}]\tlr: {cos_decay.get_last_lr()[0]:.6f}")

            self.train_mode()

            for i, data in enumerate(self.train_loader):
                x, y = data
                x, y = x.cuda(), y.cuda()

                logits = self.net(x)
                loss = self.ce(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    print(f"Ep:[{epoch + 1}/{self.args.epochs}] It: {i + 1}/{iter_per_epoch}, loss:{loss:.4f}")

            torch.save(self.net.state_dict(), os.path.join(self.args.model_path, self.args.model_name))
            test_acc = self.test(train=(epoch + 1) % 25 == 0)

            best_acc = max(best_acc, test_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

    def compute_test_metric(self, loader):
        self.net.eval()

        actual = []
        all_logits = []

        for data in loader:
            x, y = data
            x = x.cuda()

            with torch.no_grad():
                logits = self.net(x)

            actual.append(y)
            all_logits.append(logits.cpu())

        actual = torch.cat(actual)

        all_logits = torch.cat(all_logits)
        predictions = torch.max(all_logits, dim=-1)[1]

        acc = accuracy_score(y_true=actual, y_pred=predictions)
        loss = self.ce(all_logits, actual)
        cm = confusion_matrix(y_true=actual, y_pred=predictions, labels=range(self.args.n_classes))

        return acc, loss, cm

    def test(self, train=False):
        if train:
            acc, loss, cm = self.compute_test_metric(self.train_loader)
            print(f"Train Accuracy: {acc:.2%}\tLoss: {loss:.2f}")
            if self.args.cm:
                print(cm)

        acc, loss, cm = self.compute_test_metric(self.test_loader)
        print(f"Test Accuracy: {acc:.2%}\tLoss: {loss:.2f}")
        if self.args.cm:
            print(cm)
        return acc
