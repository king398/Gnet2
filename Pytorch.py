import os
import json
import random
import collections

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange, tqdm


def convert_image_id_2_path(image_id: str, is_train: bool = True) -> str:
	folder = "Train" if is_train else "test"
	return "/content/{}/{}/{}/{}/{}.npy".format(
		folder, image_id[0], image_id[1], image_id[2], image_id
	)


train_df = pd.read_csv("/content/Train/ing_labels.csv")
train_df
import torch
from nnAudio.Spectrogram import CQT1992v2

Q_TRANSFORM = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=32)
from scipy import signal


def visualize_sample_qtransform(
		_id,
		target,
		signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo"),
		sr=2048,
):
	x = np.load(convert_image_id_2_path(_id))
	plt.figure(figsize=(16, 5))
	for i in range(3):
		waves = x[i] / np.max(x[i])
		waves = waves
		waves = torch.from_numpy(waves).float()
		image = Q_TRANSFORM(waves)

		plt.subplot(1, 3, i + 1)
		plt.imshow(image.squeeze())
		plt.title(signal_names[i], fontsize=14)

	plt.suptitle(f"id: {_id} target: {target}", fontsize=16)
	plt.show()


from sklearn.metrics import roc_auc_score, roc_curve, auc

list_y_true = [
	[1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
	[1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
	[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],  # IMBALANCE
]
list_y_pred = [
	[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
	[0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.5],
	[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],  # IMBALANCE
]

for y_true, y_pred in zip(list_y_true, list_y_pred):
	fpr, tpr, _ = roc_curve(y_true, y_pred)
	roc_auc = auc(fpr, tpr)

	plt.figure(figsize=(5, 5))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

submission = pd.read_csv("/content/sample_submission.csv")
submission.to_csv("submission.csv", index=False)
import time

import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
from torch.autograd import Variable
import efficientnet_pytorch


def set_seed(seed):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True


set_seed(42)
from scipy import signal


class DataRetriever(torch_data.Dataset):
	def __init__(self, paths, targets):
		self.paths = paths
		self.targets = targets

		self.q_transform = CQT1992v2(
			sr=2048, fmin=20, fmax=1024, hop_length=32
		)

	def __len__(self):
		return len(self.paths)

	def __get_qtransform(self, x):
		image = []
		for i in range(3):
			waves = x[i] / np.max(x[i])
			waves = torch.from_numpy(waves).float()
			channel = self.q_transform(waves).squeeze().numpy()
			image.append(channel)

		return torch.tensor(image).float()

	def __getitem__(self, index):
		file_path = convert_image_id_2_path(self.paths[index])
		x = np.load(file_path)
		image = self.__get_qtransform(x)

		y = torch.tensor(self.targets[index], dtype=torch.float)

		return {"X": image, "y": y}


df_train, df_valid = sk_model_selection.train_test_split(
	train_df,
	test_size=0.2,
	random_state=42,
	stratify=train_df["target"],
)
train_data_retriever = DataRetriever(
	df_train["id"].values,
	df_train["target"].values,
)

valid_data_retriever = DataRetriever(
	df_valid["id"].values,
	df_valid["target"].values,
)
train_loader = torch_data.DataLoader(
	train_data_retriever,
	batch_size=124,
	shuffle=True,
	num_workers=4,
)

valid_loader = torch_data.DataLoader(
	valid_data_retriever,
	batch_size=124,
	shuffle=False,
	num_workers=4,
)
import timm

import efficientnet_pytorch


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
		n_features = self.net._fc.in_features
		self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

	def forward(self, x):
		out = self.net(x)
		return out


class LossMeter:
	def __init__(self):
		self.avg = 0
		self.n = 0

	def update(self, val):
		self.n += 1
		# incremental update
		self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
	def __init__(self):
		self.avg = 0
		self.n = 0

	def update(self, y_true, y_pred):
		y_true = y_true.cpu().numpy().astype(int)
		y_pred = y_pred.cpu().numpy() >= 0
		last_n = self.n
		self.n += len(y_true)
		true_count = np.sum(y_true == y_pred)
		# incremental update
		self.avg = true_count / self.n + last_n / self.n * self.avg


class Trainer:
	def __init__(
			self,
			model,
			device,
			optimizer,
			criterion,
			loss_meter,
			score_meter
	):
		self.model = model
		self.device = device
		self.optimizer = optimizer
		self.criterion = criterion
		self.loss_meter = loss_meter
		self.score_meter = score_meter

		self.best_valid_score = -np.inf
		self.n_patience = 0

		self.messages = {
			"epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
			"checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
			"patience": "\nValid score didn't improve last {} epochs."
		}

	def fit(self, epochs, train_loader, valid_loader, save_path, patience):
		for n_epoch in range(1, epochs + 1):
			self.info_message("EPOCH: {}", n_epoch)

			train_loss, train_score, train_time = self.train_epoch(train_loader)
			valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader)

			self.info_message(
				self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_time
			)

			self.info_message(
				self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_time
			)

			if True:
				#             if self.best_valid_score < valid_score:
				self.info_message(
					self.messages["checkpoint"], self.best_valid_score, valid_score, save_path
				)
				self.best_valid_score = valid_score
				self.save_model(n_epoch, save_path)
				self.n_patience = 0
			else:
				self.n_patience += 1

			if self.n_patience >= patience:
				self.info_message(self.messages["patience"], patience)
				break

	def train_epoch(self, train_loader):
		self.model.train()
		t = time.time()
		train_loss = self.loss_meter()
		train_score = self.score_meter()

		for step, batch in tqdm(enumerate(train_loader, 1)):
			X = batch["X"].to(self.device)
			targets = batch["y"].to(self.device)
			self.optimizer.zero_grad()
			outputs = self.model(X).squeeze(1)

			loss = self.criterion(outputs, targets)
			loss.backward()

			train_loss.update(loss.detach().item())
			train_score.update(targets, outputs.detach())

			self.optimizer.step()

			_loss, _score = train_loss.avg, train_score.avg
			message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}'
			self.info_message(message, step, len(train_loader), _loss, _score, end="\r")

		return train_loss.avg, train_score.avg, int(time.time() - t)

	def valid_epoch(self, valid_loader):
		self.model.eval()
		t = time.time()
		valid_loss = self.loss_meter()
		valid_score = self.score_meter()

		for step, batch in tqdm(enumerate(valid_loader, 1)):
			with torch.no_grad():
				X = batch["X"].to(self.device)
				targets = batch["y"].to(self.device)

				outputs = self.model(X).squeeze(1)
				loss = self.criterion(outputs, targets)

				valid_loss.update(loss.detach().item())
				valid_score.update(targets, outputs)

			_loss, _score = valid_loss.avg, valid_score.avg
			message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}'
			self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")

		return valid_loss.avg, valid_score.avg, int(time.time() - t)

	def save_model(self, n_epoch, save_path):
		torch.save(
			{
				"model_state_dict": self.model.state_dict(),
				"optimizer_state_dict": self.optimizer.state_dict(),
				"best_valid_score": self.best_valid_score,
				"n_epoch": n_epoch,
			},
			save_path,
		)

	@staticmethod
	def info_message(message, *args, end="\n"):
		print(message.format(*args), end=end)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch_functional.binary_cross_entropy_with_logits

trainer = Trainer(
	model,
	device,
	optimizer,
	criterion,
	LossMeter,
	AccMeter
)

history = trainer.fit(
	5,
	train_loader,
	valid_loader,
	"best-model.pth",
	100,
)
checkpoint = torch.load("best-model.pth")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval();


class DataRetriever(torch_data.Dataset):
	def __init__(self, paths):
		self.paths = paths

		self.q_transform = CQT1992v2(
			sr=2048, fmin=20, fmax=1024, hop_length=32
		)

	def __len__(self):
		return len(self.paths)

	def __get_qtransform(self, x):
		image = []
		for i in range(3):
			waves = x[i] / np.max(x[i])
			waves = torch.from_numpy(waves).float()
			channel = self.q_transform(waves).squeeze().numpy()
			image.append(channel)

		return torch.tensor(image).float()

	def __getitem__(self, index):
		file_path = convert_image_id_2_path(self.paths[index], is_train=False)
		x = np.load(file_path)
		image = self.__get_qtransform(x)

		return {"X": image, "id": self.paths[index]}


test_data_retriever = DataRetriever(
	submission["id"].values,
)

test_loader = torch_data.DataLoader(
	test_data_retriever,
	batch_size=32,
	shuffle=False,
	num_workers=8,
)
y_pred = []
ids = []

for e, batch in tqdm(enumerate(test_loader)):
	print(f"{e}/{len(test_loader)}", end="\r")
	with torch.no_grad():
		y_pred.extend(torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze())
		ids.extend(batch["id"])
submission = pd.DataFrame({"id": ids, "target": y_pred})
submission.to_csv("model_submission.csv", index=False)
