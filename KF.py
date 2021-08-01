import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers as L
import math
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from random import shuffle
from sklearn.model_selection import KFold
import tensorflow_addons as tfa
import albumentations as A
from sklearn.model_selection import KFold

train_labels = pd.read_csv('/content/Train/ing_labels.csv')
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

path = list(train_labels['id'])
for i in range(len(path)):
	path[i] = '/content/Train/' + path[i][0] + '/' + path[i][1] + '/' + path[i][2] + '/' + path[i] + '.npy'


def mixup(image, label, PROBABILITY=1.0):
	# input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
	# output - a batch of images with mixup applied

	imgs = [];
	labs = []
	for j in range(len(image)):
		# DO MIXUP WITH PROBABILITY DEFINED ABOVE
		P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
		# CHOOSE RANDOM
		k = tf.cast(tf.random.uniform([], 0, len(image)), tf.int32)
		a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
		# MAKE MIXUP IMAGE
		img1 = image[j,]
		img2 = image[k,]
		imgs.append((1 - a) * img1 + a * img2)
		# MAKE CUTMIX LABEL
		lab1 = label[j]
		lab2 = label[k]
		labs.append((1 - a) * lab1 + a * lab2)

	# RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
	image2 = tf.reshape(tf.stack(imgs), (len(image), 69, 193, 1))
	label2 = tf.reshape(tf.stack(labs), (len(image)))

	return image2, label2


def id2path(idx, is_train=True):
	path = '../input/g2net-gravitational-wave-detection'
	if is_train:
		path = "/content/Train/" + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	else:
		path += '/test/' + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	return path


import torch
from nnAudio.Spectrogram import CQT1992v2


def increase_dimension(idx, is_train, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024,
                                                          hop_length=32)):  # in order to use efficientnet we need 3 dimension images
	waves = np.load(id2path(idx, is_train))
	waves = np.hstack(waves)
	waves = waves / np.max(waves)
	waves = torch.from_numpy(waves).float()
	image = transform(waves)
	image = np.array(image)
	image = np.transpose(image, (1, 2, 0))
	return image


example = np.load(path[4])
fig, a = plt.subplots(3, 1)
a[0].plot(example[1], color='green')
a[1].plot(example[1], color='red')
a[2].plot(example[1], color='yellow')
fig.suptitle('Target 1', fontsize=16)
plt.show()
plt.show()


class Dataset(Sequence):
	def __init__(self, idx, y=None, batch_size=256, shuffle=True, valid=False):
		self.idx = idx
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.valid = valid
		if y is not None:
			self.is_train = True
		else:
			self.is_train = False
		self.y = y

	def __len__(self):
		return math.ceil(len(self.idx) / self.batch_size)

	def __getitem__(self, ids):
		batch_ids = self.idx[ids * self.batch_size:(ids + 1) * self.batch_size]
		if self.y is not None:
			batch_y = self.y[ids * self.batch_size: (ids + 1) * self.batch_size]

		list_x = np.array([increase_dimension(x, self.is_train) for x in batch_ids])
		batch_X = np.stack(list_x)
		batch_X = tf.image.resize(images=batch_X, size=(69, 193))

		if self.valid == False:
			batch_X, batch_y = mixup(np.array(batch_X), np.array(batch_y))

		if self.is_train:
			return np.array(batch_X), batch_y
		else:
			return batch_X

	def on_epoch_end(self):
		if self.shuffle and self.is_train:
			ids_y = list(zip(self.idx, self.y))
			shuffle(ids_y)
			self.idx, self.y = list(zip(*ids_y))


train_idx = train_labels['id'].values
y = train_labels['target'].values


def model():
	import efficientnet.tfkeras as efn

	model = tf.keras.Sequential(
		[L.InputLayer(input_shape=(69, 193, 1)), L.Conv2D(3, 3, activation='relu', padding='same'),
		 efn.EfficientNetB0(include_top=False, input_shape=(), weights='imagenet'),
		 L.GlobalAveragePooling2D(),
		 L.Dense(32, activation='relu'),
		 L.Dense(1, activation='sigmoid')])
	model.summary()

	opt = tf.keras.optimizers.Adam(0.001)
	model.compile(optimizer=opt,
	              loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
	return model


oof_preds = []
skf = KFold(n_splits=5, random_state=42, shuffle=True)
fold = 0
for train, test in skf.split(train_idx, y=y):
	tf.keras.backend.clear_session()
	train_dataset = Dataset(train_idx[train], y[train])
	test_dataset = Dataset(train_idx[test], y[test], valid=True)
	model = model()
	if fold == 0 or fold == 1 or fold == 2:
		best = tf.keras.callbacks.ModelCheckpoint("/content/Temp", monitor="val_auc", save_best_only=True)

		history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=best)
		model.save("/content/Models/Fold" + str(fold))

		oof_preds.append(history.history['val_auc'])
