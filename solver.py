import torch
import torch.nn as nn
import os
from torch import optim
from model import encoder, classifier, prot_classifier
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
from utils import adjust_learning_rate
from data_loader import prot_loader, cls_loader


class Solver(object):
	def __init__(self, args):
		self.args = args

		self.prot_train_loader, self.prot_test_loader = prot_loader(self.args)
		self.cls_train_loader, self.cls_test_loader = cls_loader(self.args)

		self.enc = encoder(args).cuda()
		self.prot_clf = prot_classifier(args).cuda()
		self.cls_clf = classifier(args).cuda()

		if self.args.method == 'finetune':
			self.enc.load_state_dict(torch.load(os.path.join(self.args.model_path, 'pr_enc.pkl')))

		self.ce = nn.CrossEntropyLoss()
		self.bce = nn.BCEWithLogitsLoss()

		self.flag = True

	def freeze_enc(self):
		ct = 0
		frozen = 0
		if self.flag:
			print("Frozen layers:")

		layer1 = self.enc.encoder[0] #self.enc.encoder.embeddings
		if ct < self.args.num_layers_to_freeze:
			layer1.eval()
			for param in layer1.parameters():
				param.requires_grad = False
			if self.flag:
				frozen += 1
				print(layer1, end="\n\n")
		else:
			layer1.train()
			for param in layer1.parameters():
				param.requires_grad = True
		ct = ct + 1

		layer2 = self.enc.encoder[1].layer #self.enc.encoder.encoder.layer
		for child in layer2.children():
			if ct < self.args.num_layers_to_freeze:
				child.eval()
				for param in child.parameters():
					param.requires_grad = False            
				if self.flag:
					frozen += 1
					print(child, end="\n\n")
			else:
				child.train()
				for param in child.parameters():
					param.requires_grad = True
			ct = ct + 1       

		layer3, layer4 = self.enc.encoder[3], self.enc.fc
		if ct < self.args.num_layers_to_freeze:
			layer3.eval()
			for param in layer3.parameters():
				param.requires_grad = False
			layer4.eval()
			for param in layer4.parameters():
				param.requires_grad = False
			if self.flag:
				frozen += 1
				print(layer3, "\n", layer4, "\n")
		else:
			layer3.train()
			for param in layer3.parameters():
				param.requires_grad = True
			layer3.train()
			for param in layer4.parameters():
				param.requires_grad = True
		if self.flag:
			self.flag = False
			print(f"Total frozen layers: {frozen}")

		layer1.cls_token.requires_grad = True

	def prot_train(self):
		iter_per_epoch = len(self.prot_train_loader)
		print(f"Iters per epoch: {iter_per_epoch:d}")

		optimizer = optim.AdamW(list(self.enc.parameters()) + list(self.prot_clf.parameters()), self.args.lr, weight_decay=self.args.weight_decay)

		total_iters = 0
		best_acc = 0

		for epoch in range(self.args.pr_epochs):
			new_lr = adjust_learning_rate(self.args, optimizer, epoch, self.args.pr_epochs)
			print(f"\nEp:[{epoch+1}/{self.args.pr_epochs}]\tlr:{new_lr}")

			self.enc.train()
			self.prot_clf.train()

			for i, data in enumerate(self.prot_train_loader):
				total_iters += 1

				imgs, prot_labels = data
				imgs, prot_labels = imgs.cuda(), prot_labels.cuda()
				imgs, prot_labels = imgs.flatten(0, 1), prot_labels.flatten(0, 1)

				feats = self.enc(imgs, prot=True)
				class_out = self.prot_clf(feats)

				loss = self.ce(class_out.reshape([-1, 4]), prot_labels.reshape([-1]))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if i % 50 == 0 or i == (iter_per_epoch - 1):
					print(f"It: {i+1}/{iter_per_epoch}\t t_it: {total_iters}\terr:{loss:.4f}")

			test_acc = self.prot_test(train=False)

			if test_acc > best_acc:
				best_acc = test_acc
				torch.save(self.enc.state_dict(), os.path.join(self.args.model_path, 'pr_enc.pkl'))
				torch.save(self.prot_clf.state_dict(), os.path.join(self.args.model_path, 'pr_clf.pkl'))

			print(f"Best test acc: {best_acc:.2%}")

		self.enc.load_state_dict(torch.load(os.path.join(self.args.model_path, 'pr_enc.pkl')))
		self.prot_clf.load_state_dict(torch.load(os.path.join(self.args.model_path, 'pr_clf.pkl')))

	def compute_prot_metrics(self, loader):
		self.enc.eval()
		self.cls_clf.eval()

		actual = []
		predictions = []

		for data in loader:
			imgs, pr_labels = data
			imgs = imgs.cuda()
			imgs, pr_labels = imgs.flatten(0, 1), pr_labels.flatten(0, 1)

			with torch.no_grad():
				feats = self.enc(imgs, prot=True)
				class_out = self.prot_clf(feats)

			actual.append(pr_labels)
			predictions.append(class_out.cpu())

		actual = torch.cat(actual)
		predictions = torch.cat(predictions)

		loss = self.ce(predictions.reshape(-1, 4), actual.reshape(-1))

		predictions_label = predictions.max(-1)[1]
		i_idx = actual[:, 0] != -100
		iacc = (predictions_label[i_idx, 0] == actual[i_idx, 0]).float().mean()
		pacc = (predictions_label[~i_idx, 1:] == actual[~i_idx, 1:]).float().mean()

		actual = actual.flatten()
		predictions_label = predictions_label.flatten()
		valid_idx = actual != -100
		acc = accuracy_score(actual[valid_idx], predictions_label[valid_idx])

		return iacc, pacc, acc, loss

	def prot_test(self, train=False):

		if train:
			iacc, pacc, acc, loss = self.compute_prot_metrics(self.prot_train_loader)
			print(f"Train accuracy Image: {iacc:.2%}\tPatch: {pacc:.2%}\tOverall: {acc:.2%}\tLoss: {loss:.4f}")

		iacc, pacc, acc, loss = self.compute_prot_metrics(self.prot_test_loader)
		print(f"Test accuracy Image: {iacc:.2%}\tPatch: {pacc:.2%}\tOverall: {acc:.2%}\tLoss: {loss:.4f}")

		return acc

	def cls_finetune(self):
		iter_per_epoch = len(self.cls_train_loader)
		print(f"Iters per epoch: f{iter_per_epoch}")

		optimizer = optim.AdamW(list(self.enc.parameters()) + list(self.cls_clf.parameters()), self.args.lr, weight_decay=self.args.weight_decay)

		total_iters = 0
		best_acc = 0

		for epoch in range(self.args.ft_epochs):
			self.freeze_enc()
			self.cls_clf.train()

			new_lr = adjust_learning_rate(self.args, optimizer, epoch, self.args.ft_epochs)
			print(f"\nEp:[{epoch+1}/{self.args.ft_epochs}]\tlr:{new_lr}")

			for i, data in enumerate(self.cls_train_loader):
				total_iters += 1

				imgs, labels = data
				imgs, labels = imgs.cuda(), labels.cuda()

				feats = self.enc(imgs, prot=False)
				class_out = self.cls_clf(feats)

				loss = self.ce(class_out, labels)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if i % 50 == 0 or i == (iter_per_epoch - 1):
					print(f'It: {i+1}/{iter_per_epoch}\tt_it: {total_iters}\tloss:{loss.item():.4f}')

			test_acc = self.cls_test()

			if test_acc > best_acc:
				best_acc = test_acc
				torch.save(self.enc.state_dict(), os.path.join(self.args.model_path, 'src_enc_'+str(self.args.num_layers_to_freeze)+'.pkl'))
				torch.save(self.cls_clf.state_dict(), os.path.join(self.args.model_path, 'src_clf_'+str(self.args.num_layers_to_freeze)+'.pkl'))

			print(f"Best test acc: {best_acc:.2%}")

		self.enc.load_state_dict(torch.load(os.path.join(self.args.model_path, 'src_enc_'+str(self.args.num_layers_to_freeze)+'.pkl')))
		self.cls_clf.load_state_dict(torch.load(os.path.join(self.args.model_path, 'src_clf_'+str(self.args.num_layers_to_freeze)+'.pkl')))

	def compute_cls_metric(self, loader):
		self.enc.eval()
		self.cls_clf.eval()

		actual = []
		predictions = []

		for data in loader:
			imgs, labels = data
			imgs = imgs.cuda()

			with torch.no_grad():
				feats = self.enc(imgs, prot=False)
				class_out = self.cls_clf(feats)

			actual.append(labels)
			predictions.append(class_out.cpu())

		actual = torch.cat(actual)
		predictions = torch.cat(predictions)

		acc1 = top_k_accuracy_score(actual, predictions, k=1)
		acc5 = top_k_accuracy_score(actual, predictions, k=5)
		loss = self.ce(predictions, actual)
		pred = torch.max(predictions, dim=-1)[1]
		cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.num_classes))

		return acc1, acc5, loss, cm

	def cls_test(self, train=False):

		if train:
			acc1, acc5, loss, cm = self.compute_cls_metric(self.cls_train_loader)
			print(f"Train Accuracy Top-1: {acc1:.2%} Top-5: {acc5:.2%}\tLoss: {loss:.2f}")
			if self.args.cm:
				print(cm)

		acc1, acc5, loss, cm = self.compute_cls_metric(self.cls_test_loader)
		print(f"Test Accuracy Top-1: {acc1:.2%} Top-5: {acc5:.2%}\tLoss: {loss:.2f}")
		if self.args.cm:
			print(cm)

		return acc1
