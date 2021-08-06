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

	inp = inp = tf.keras.layers.Input(shape=(69, 193, 1))
	base = L.Conv2D(3, 3, activation='relu', padding='same')
	x = base(inp)
	x = efn.EfficientNetB7(include_top=False, input_shape=(), weights='imagenet')(x)
	x = L.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	model = tf.keras.Model(inputs=inp, outputs=x)
	lr_decayed_fn = tf.keras.experimental.CosineDecay(
		1e-3,
		820,
	)

	opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=1e-4)
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
	if fold == 2 or fold == 3:
		model = model()

		best = tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_auc:.4f}' + ":" + str(fold),
		                                          monitor="val_auc")

		history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=best)
		model = tf.keras.models.load_model("/content/Temp")
		model.save("/content/Models/Fold" + str(fold))

		oof_preds.append(history.history['val_auc'])
	fold += 1
