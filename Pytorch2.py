import os
from albumentations.augmentations.transforms import ToFloat
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns

train = pd.read_csv('/content/Train/ing_labels.csv')
test = pd.read_csv('/content/sample_submission.csv')


def get_train_file_path(image_id):
	return "/content/Train/{}/{}/{}/{}.npy".format(
		image_id[0], image_id[1], image_id[2], image_id)


def get_test_file_path(image_id):
	return "/content/Test/{}/{}/{}/{}.npy".format(
		image_id[0], image_id[1], image_id[2], image_id)


train['file_path'] = train['id'].apply(get_train_file_path)
test['file_path'] = test['id'].apply(get_test_file_path)
import torch
from nnAudio.Spectrogram import CQT1992v2


def apply_qtransform(waves, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)):
	waves = np.hstack(waves)
	waves = waves / np.max(waves)
	waves = torch.from_numpy(waves).float()
	image = transform(waves)
	return image


def apply_qtransform1(waves, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)):
	waves = np.hstack(waves)
	waves = waves / np.max(waves)
	waves = filterSig(waves)
	waves = torch.from_numpy(waves).float()
	image = transform(waves)
	return image


from scipy import signal

bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)


def filterSig(wave, a=aHP, b=bHP):
	'''Apply a 20Hz high pass filter to the three events'''
	return np.array(signal.filtfilt(b, a, wave))


for i in range(1):
	waves = filterSig(np.load(train.loc[i, 'file_path']))
	image = apply_qtransform(waves)
	target = train.loc[i, 'target']
	plt.imshow(image[0])
	plt.title(f"target: {target}")
	plt.show()
for i in range(1):
	waves = np.load(train.loc[i, 'file_path'])
	image = apply_qtransform(waves)
	target = train.loc[i, 'target']
	plt.imshow(image[0])
	plt.title(f"target: {target}")
	plt.show()
for i in range(1):
	waves = filterSig(np.load(train.loc[i, 'file_path']))
	image = apply_qtransform1(waves)
	target = train.loc[i, 'target']
	plt.imshow(image[0])
	plt.title(f"target: {target}")
	plt.show()
train['target'].hist()
# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)


class CFG:
	apex = True
	debug = False
	print_freq = 100
	num_workers = 4
	model_name = 'tf_efficientnet_b7_ns'
	scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
	epochs = 3
	# factor=0.2 # ReduceLROnPlateau
	# patience=4 # ReduceLROnPlateau
	# eps=1e-6 # ReduceLROnPlateau
	T_max = 3  # CosineAnnealingLR
	# T_0=3 # CosineAnnealingWarmRestarts
	lr = 1e-4
	min_lr = 1e-6
	batch_size = 384
	weight_decay = 1e-6
	gradient_accumulation_steps = 1
	max_grad_norm = 1000
	qtransform_params = {"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
	seed = 42
	target_size = 1
	target_col = 'target'
	n_fold = 5
	trn_fold = [0]  # [0, 1, 2, 3, 4]
	train = True
	grad_cam = True


if CFG.debug:
	CFG.epochs = 1
	train = train.sample(n=10000, random_state=CFG.seed).reset_index(drop=True)
# ====================================================
# Library
# ====================================================
import sys

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

import timm

from torch.cuda.amp import autocast, GradScaler

import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
	score = roc_auc_score(y_true, y_pred)
	return score


def init_logger(log_file=OUTPUT_DIR + 'train.log'):
	from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
	logger = getLogger(__name__)
	logger.setLevel(INFO)
	handler1 = StreamHandler()
	handler1.setFormatter(Formatter("%(message)s"))
	handler2 = FileHandler(filename=log_file)
	handler2.setFormatter(Formatter("%(message)s"))
	logger.addHandler(handler1)
	logger.addHandler(handler2)
	return logger


LOGGER = init_logger()


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG.seed)
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_col])):
	train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)


# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
	def __init__(self, df, transform=None):
		self.df = df
		self.file_names = df['file_path'].values
		self.labels = df[CFG.target_col].values
		self.wave_transform = CQT1992v2(**CFG.qtransform_params)
		self.transform = transform

	def __len__(self):
		return len(self.df)

	def apply_qtransform(self, waves, transform):
		waves = np.hstack(waves)
		waves = filterSig(waves)
		waves = waves / np.max(waves)
		waves = torch.from_numpy(waves).float()
		image = transform(waves)
		return image

	def __getitem__(self, idx):
		file_path = self.file_names[idx]
		waves = np.load(file_path)
		image = self.apply_qtransform(waves, self.wave_transform)
		image = image.squeeze().numpy()
		if self.transform:
			image = self.transform(image=image)['image']
		label = torch.tensor(self.labels[idx]).float()
		return image, label


def get_transforms(*, data):
	if data == 'train':
		return A.Compose([

			ToTensorV2(),
		])

	elif data == 'valid':
		return A.Compose([
			A.ToFloat(),

			ToTensorV2(),
		])


# ====================================================
# MODEL
# ====================================================


class CustomModel(nn.Module):
	def __init__(self, cfg, pretrained=False):
		super().__init__()
		self.cfg = cfg
		self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=1)
		self.n_features = self.model.classifier.in_features
		self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)

	def forward(self, x):
		output = self.model(x)
		return output

bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)


def filterSig(wave, a=aHP, b=bHP):
	'''Apply a 20Hz high pass filter to the three events'''
	return np.array(signal.filtfilt(b, a, wave))
class GradCAMDataset(Dataset):
	def __init__(self, df):
		self.df = df
		self.image_ids = df['id'].values
		self.file_names = df['file_path'].values
		self.labels = df[CFG.target_col].values
		self.wave_transform = CQT1992v2(**CFG.qtransform_params)
		self.transform = get_transforms(data='valid')

	def __len__(self):
		return len(self.df)

	def apply_qtransform(self, waves, transform):
		waves = np.hstack(waves)
		waves = filterSig(waves)
		waves = waves / np.max(waves)
		waves = torch.from_numpy(waves).float()
		image = transform(waves)
		return image

	def __getitem__(self, idx):
		image_id = self.image_ids[idx]
		file_path = self.file_names[idx]
		waves = filterSig(np.load(file_path))
		image = self.apply_qtransform(waves, self.wave_transform)
		image = image.squeeze().numpy()
		vis_image = image.copy()
		if self.transform:
			image = self.transform(image=image)['image']
		label = torch.tensor(self.labels[idx]).float()
		return image_id, image, vis_image, label


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
	if CFG.apex:
		scaler = GradScaler()
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	scores = AverageMeter()
	# switch to train mode
	model.train()
	start = end = time.time()
	global_step = 0
	for step, (images, labels) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		images = images.to(device)
		labels = labels.to(device)
		batch_size = labels.size(0)
		if CFG.apex:
			with autocast():
				y_preds = model(images)
				loss = criterion(y_preds.view(-1), labels)
		else:
			y_preds = model(images)
			loss = criterion(y_preds.view(-1), labels)
		# record loss
		losses.update(loss.item(), batch_size)
		if CFG.gradient_accumulation_steps > 1:
			loss = loss / CFG.gradient_accumulation_steps
		if CFG.apex:
			scaler.scale(loss).backward()
		else:
			loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
		if (step + 1) % CFG.gradient_accumulation_steps == 0:
			if CFG.apex:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad()
			global_step += 1
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
			print('Epoch: [{0}][{1}/{2}] '
			      'Elapsed {remain:s} '
			      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
			      'Grad: {grad_norm:.4f}  '
			      'LR: {lr:.6f}  '
			      .format(epoch + 1, step, len(train_loader),
			              remain=timeSince(start, float(step + 1) / len(train_loader)),
			              loss=losses,
			              grad_norm=grad_norm,
			              lr=scheduler.get_lr()[0]))

	return losses.avg


def valid_fn(valid_loader, model, criterion, device):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	scores = AverageMeter()
	# switch to evaluation mode
	model.eval()
	preds = []
	start = end = time.time()
	for step, (images, labels) in enumerate(valid_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		images = images.to(device)
		labels = labels.to(device)
		batch_size = labels.size(0)
		# compute loss
		with torch.no_grad():
			y_preds = model(images)
		loss = criterion(y_preds.view(-1), labels)
		losses.update(loss.item(), batch_size)
		# record accuracy
		preds.append(y_preds.sigmoid().to('cpu').numpy())
		if CFG.gradient_accumulation_steps > 1:
			loss = loss / CFG.gradient_accumulation_steps
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
			print('EVAL: [{0}/{1}] '
			      'Elapsed {remain:s} '
			      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
			      .format(step, len(valid_loader),
			              loss=losses,
			              remain=timeSince(start, float(step + 1) / len(valid_loader))))
	predictions = np.concatenate(preds)
	return losses.avg, predictions


def get_grad_cam(model, device, x_tensor, img, label, plot=False):
	result = {"vis": None, "img": None, "prob": None, "label": None}

	# model prob
	with torch.no_grad():
		prob = model(x_tensor.unsqueeze(0).to(device))
	prob = np.concatenate(prob.sigmoid().to('cpu').numpy())[0]

	# grad-cam
	target_layer = model.model.conv_head
	cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
	output = cam(input_tensor=x_tensor.unsqueeze(0))
	try:
		vis = show_cam_on_image(x_tensor.numpy().transpose((1, 2, 0)), output[0])
	except:
		return result

	# plot result
	if plot:
		fig, axes = plt.subplots(figsize=(16, 12), ncols=2)
		axes[0].imshow(vis)
		axes[0].set_title(f"prob={prob:.4f}")
		axes[1].imshow(img)
		axes[1].set_title(f"target={label}")
		plt.show()

	result = {"vis": vis, "img": img, "prob": prob, "label": label}

	return result


# ====================================================
# Train loop
# ====================================================
def train_loop(folds, fold):
	
		LOGGER.info(f"========== fold: {fold} training ==========")

		# ====================================================
		# loader
		# ====================================================
		trn_idx = folds[folds['fold'] != fold].index
		val_idx = folds[folds['fold'] == fold].index

		train_folds = folds.loc[trn_idx].reset_index(drop=True)
		valid_folds = folds.loc[val_idx].reset_index(drop=True)
		valid_labels = valid_folds[CFG.target_col].values

		train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
		valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='train'))

		train_loader = DataLoader(train_dataset,
		                          batch_size=CFG.batch_size,
		                          shuffle=True,
		                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
		valid_loader = DataLoader(valid_dataset,
		                          batch_size=CFG.batch_size * 2,
		                          shuffle=False,
		                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

		# ====================================================
		# scheduler
		# ====================================================
		def get_scheduler(optimizer):
			if CFG.scheduler == 'ReduceLROnPlateau':
				scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience,
				                              verbose=True,
				                              eps=CFG.eps)
			elif CFG.scheduler == 'CosineAnnealingLR':
				scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
			elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
				scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr,
				                                        last_epoch=-1)
			return scheduler

		# ====================================================
		# model & optimizer
		# ====================================================
		model = CustomModel(CFG, pretrained=True)
		model.to(device)

		optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
		scheduler = get_scheduler(optimizer)

		# ====================================================
		# loop
		# ====================================================
		criterion = nn.BCEWithLogitsLoss()

		best_score = 0.
		best_loss = np.inf

		for epoch in range(CFG.epochs):

			start_time = time.time()

			# train
			avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

			# eval
			avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

			if isinstance(scheduler, ReduceLROnPlateau):
				scheduler.step(avg_val_loss)
			elif isinstance(scheduler, CosineAnnealingLR):
				scheduler.step()
			elif isinstance(scheduler, CosineAnnealingWarmRestarts):
				scheduler.step()

			# scoring
			score = get_score(valid_labels, preds)

			elapsed = time.time() - start_time

			LOGGER.info(
				f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
			LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

			if score > best_score:
				best_score = score
				LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
				torch.save({'model': model.state_dict(),
				            'preds': preds},
				           OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth')

			if avg_val_loss < best_loss:
				best_loss = avg_val_loss
				LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
				torch.save({'model': model.state_dict(),
				            'preds': preds},
				           OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss.pth')

		valid_folds['preds'] = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth',
		                                  map_location=torch.device('cpu'))['preds']

		return valid_folds


# ====================================================
def main():

    """
    Prepare: 1.train 
    """

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
    
    if CFG.grad_cam:
        N = 5
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                # load model
                model = CustomModel(CFG, pretrained=False)
                state = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth', 
                                   map_location=torch.device('cpu'))['model']
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                # load oof
                oof = pd.read_csv(OUTPUT_DIR+'oof_df.csv')
                oof = oof[oof['fold'] == fold].reset_index(drop=True)
                # grad-cam (oof ascending=False)
                count = 0
                oof = oof.sort_values('preds', ascending=False)
                valid_dataset = GradCAMDataset(oof)
                for i in range(len(valid_dataset)):
                    image_id, x_tensor, img, label = valid_dataset[i]
                    result = get_grad_cam(model, device, x_tensor, img, label, plot=True)
                    if result["vis"] is not None:
                        count += 1

                    if count >= N:
                        break
                # grad-cam (oof ascending=True)
                count = 0
                oof = oof.sort_values('preds', ascending=True)
                valid_dataset = GradCAMDataset(oof)
                for i in range(len(valid_dataset)):
                    image_id, x_tensor, img, label = valid_dataset[i]
                    result = get_grad_cam(model, device, x_tensor, img, label, plot=True)
                    if result["vis"] is not None:
                        count += 1

                    if count >= N:
                        break
    
if __name__ == '__main__':
    main()