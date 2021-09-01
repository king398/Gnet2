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
from CWT.cwt import ComplexMorletCWT
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


cwt_transform = ComplexMorletCWT(wavelet_width=8, fs=2048, lower_freq=20, upper_freq=500, n_scales=image_size[0],
                                 stride=int(np.ceil(4096 / image_size[0])), output='magnitude',
                                 data_format='channels_first')

from scipy import signal

bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)


def filterSig(wave, a=aHP, b=bHP):
	'''Apply a 20Hz high pass filter to the three events'''
	return np.array([signal.filtfilt(b, a, wave)])  # lfilter introduces a larger spike around 20hz


def increase_dimension(idx, is_train):
	# in order to use efficientnet we need 3 dimension images
	wavess = np.load(id2path(idx, is_train)).astype(np.float32)

	waves_f = []
	cwts_f = []  # in order to use efficientnet we need 3 dimension images
	for i in range(3):
		wave = wavess[i]  # for demonstration we will use only signal from one detector
		wave *= 1.3e+22  # normalization

		# With a filter
		wave_f = wave * signal.tukey(4096, 0.2)
		wave_f = signal.filtfilt(bHP, aHP, wave_f)
		waves_f.append(wave_f)
		wave_t_f = tf.convert_to_tensor(wave_f[np.newaxis, np.newaxis, :])
		cwt_f = cwt_transform(wave_t_f)
		cwts_f.append(np.squeeze(cwt_f.numpy()))

	return np.transpose(np.stack(cwts_f)).astype(np.float16)


class Dataset(Sequence):
	def __init__(self, idx, y=None, batch_size=64, shuffle=True, valid=False):
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

model = tf.keras.Sequential([

	efn.EfficientNetB7(include_top=False, input_shape=(), weights='imagenet'),
	L.GlobalAveragePooling2D(),
	L.Dense(32, activation='relu'),
	L.Dense(1, activation='softmax', dtype=np.float16)])
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
