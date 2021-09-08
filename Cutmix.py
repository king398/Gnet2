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
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import albumentations as A
from scipy.fft import fft
from scipy import signal

train_labels = pd.read_csv('/content/Train/ing_labels.csv')
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

path = list(train_labels['id'])
for i in range(len(path)):
	path[i] = '/content/Train/' + path[i][0] + '/' + path[i][1] + '/' + path[i][2] + '/' + path[i] + '.npy'
image_size = [256, 256]


def id2path(idx, is_train=True):
	path = '../input/g2net-gravitational-wave-detection'
	if is_train:
		path = "/content/Train/" + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	else:
		path += '/test/' + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	return path


from scipy import signal

bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)


def filterSig(wave, a=aHP, b=bHP):
	'''Apply a 20Hz high pass filter to the three events'''
	return np.array([signal.filtfilt(b, a, wave)])  # lfilter introduces a larger spike around 20hz


import torch
from nnAudio.Spectrogram import CQT1992v2


def increase_dimension(idx, is_train, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024,
                                                          hop_length=4)):  # in order to use efficientnet we need 3 dimension images
	waves = np.load(id2path(idx, is_train))
	waves = np.hstack(waves)
	waves = waves / np.max(waves)
	waves = filterSig(waves)
	waves = torch.from_numpy(waves).float()
	image = transform(waves)
	image = np.array(image)
	image = np.transpose(image, (1, 2, 0))
	return image


class Dataset(Sequence):
	def __init__(self, idx, y=None, batch_size=24, shuffle=True, valid=False):
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

		batch_X = np.array([increase_dimension(x, self.is_train) for x in batch_ids])
		batch_X = tf.image.resize(batch_X, size=(256, 256))

		if self.is_train:
			return np.array(batch_X, dtype=np.float32), batch_y
		else:
			return batch_X

	def on_epoch_end(self):
		if self.shuffle and self.is_train:
			ids_y = list(zip(self.idx, self.y))
			shuffle(ids_y)
			self.idx, self.y = list(zip(*ids_y))


train_idx = train_labels['id'].values
y = train_labels['target'].values
x_train, x_valid, y_train, y_valid = train_test_split(train_idx, y, test_size=0.2, random_state=42, stratify=y)
train_dataset = Dataset(x_train, y_train)
valid_dataset = Dataset(x_valid, y_valid, valid=True)
import efficientnet.tfkeras as efn


def build_model(size=256, efficientnet_size=0, weights="imagenet", count=0):
	inputs = tf.keras.layers.Input(shape=(size, size, 3))

	efn_string = f"EfficientNetB{efficientnet_size}"
	efn_layer = getattr(efn, efn_string)(input_shape=(size, size, 3), weights=weights, include_top=False)

	x = L.Conv2D(3, 3, activation='relu', padding='same')(inputs)
	x  = efn_layer(x)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)

	x = tf.keras.layers.Dropout(0.2)(x)
	x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
	model = tf.keras.Model(inputs=inputs, outputs=x)

	lr_decayed_fn = tf.keras.experimental.CosineDecay(1e-3, 820)
	opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=1e-4)
	loss = tf.keras.losses.BinaryCrossentropy()
	model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
	return model


model = build_model()
best = tf.keras.callbacks.ModelCheckpoint("/content/Temp", monitor="val_auc", save_best_only=True)
lr_decayed_fn = tf.keras.experimental.CosineDecay(
	1e-3,
	700,
)

opt = tf.keras.optimizers.Adam(0.001)
i = 0
print(len(train_dataset))

model.compile(optimizer=opt,
              loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
model.fit(train_dataset, epochs=5, validation_data=valid_dataset, callbacks=best)
