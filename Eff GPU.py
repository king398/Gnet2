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
import cv2
from scipy import signal
import tensorflow.keras.layers as L

train_labels = pd.read_csv('/content/Train/ing_labels.csv')
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

path = list(train_labels['id'])
for i in range(len(path)):
	path[i] = '/content/Train/' + path[i][0] + '/' + path[i][1] + '/' + path[i][2] + '/' + path[i] + '.npy'


def id2path(idx, is_train=True):
	path = '../input/g2net-gravitational-wave-detection'
	if is_train:
		path = "/content/Train/" + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	else:
		path += '/test/' + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	return path


import torch
from nnAudio.Spectrogram import CQT1992v2


def increase_dimension(idx, is_train):  # in order to use efficientnet we need 3 dimension images
	waves = np.load(id2path(idx, is_train))
	waves = np.hstack(waves)
	waves = filterSig(waves)
	waves = waves / np.max(waves)

	return waves




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

		if self.is_train:

			return np.array(list_x), batch_y



		else:
			return list_x

	def on_epoch_end(self):
		if self.shuffle and self.is_train:
			ids_y = list(zip(self.idx, self.y))
			shuffle(ids_y)
			self.idx, self.y = list(zip(*ids_y))


train_idx = train_labels['id'].values
y = train_labels['target'].values
import efficientnet.tfkeras as efn


def model():
	model = tf.keras.Sequential([L.InputLayer(input_shape=(3, 4096)),
	                             ComplexMorletCWT(n_scales=128, stride=128, output='magnitude',
	                                              data_format='channels_first', wavelet_width=8, fs=1.0, lower_freq=20,
	                                              upper_freq=1024),
	                             L.Permute((2, 3, 1)),
	                             efn.EfficientNetB7(include_top=False, weights='imagenet'),
	                             L.GlobalAveragePooling2D(),
	                             L.Dense(32, activation='relu'),
	                             L.Dense(1, activation='sigmoid')])

	opt = tf.keras.optimizers.Adam(0.001)
	model.compile(optimizer=opt,
	              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.AUC()])
	return model


oof_preds = []
skf = KFold(n_splits=4, random_state=42, shuffle=True)
fold = 0
for train, test in skf.split(train_idx, y=y):
	tf.keras.backend.clear_session()
	train_dataset = Dataset(train_idx[train], y[train])
	test_dataset = Dataset(train_idx[test], y[test], valid=True)
	if fold == 0:
		model = model()
		best = tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_auc:.4f}' + ":" + str(fold),
		                                          monitor="val_auc"),

		history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=best)

		oof_preds.append(history.history['val_auc'])
	fold += 1
