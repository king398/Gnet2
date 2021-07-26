import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import math
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision

sample_submission = pd.read_csv('/content/sample_submission.csv')
path = list(sample_submission['id'])
physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass
# Equivalent to the two lines above
for i in range(len(path)):
	path[i] = '/content/test/' + path[i][0] + '/' + path[i][1] + '/' + path[i][
		2] + '/' + path[i] + '.npy'


def id2path(idx, is_train=True):

	path = '/content/test/' + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	return path


import torch
from nnAudio.Spectrogram import CQT1992v2


def increase_dimension(idx, is_train, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024,
                                                          hop_length=64)):  # in order to use efficientnet we need 3 dimension images
	waves = np.load(id2path(idx, is_train))
	waves = np.hstack(waves)
	waves = waves / np.max(waves)
	waves = torch.from_numpy(waves).float()
	image = transform(waves)
	image = np.array(image)
	image = np.transpose(image, (1, 2, 0))
	return image


example = np.load(path[0])


class Dataset(Sequence):
	def __init__(self, idx, y=None, batch_size=256, shuffle=True):
		self.idx = idx
		self.batch_size = batch_size
		self.shuffle = shuffle
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
			return batch_X, batch_y
		else:
			return batch_X

	def on_epoch_end(self):
		if self.shuffle and self.is_train:
			ids_y = list(zip(self.idx, self.y))
			shuffle(ids_y)
			self.idx, self.y = list(zip(*ids_y))


test_idx = sample_submission['id'].values
test_dataset = Dataset(test_idx)

model = tf.keras.models.load_model(r"/content/Models/NewPipelineEfficinetB7Imagnet")
preds = model.predict(test_dataset, verbose=1)
preds = preds.reshape(-1)

submission = pd.DataFrame({'id': sample_submission['id'], 'target': preds})
submission.to_csv('submission.csv', index=False)
