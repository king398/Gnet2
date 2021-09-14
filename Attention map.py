import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
import tensorflow as tf
import re
import os
import numpy as np
import pandas as pd
from scipy.signal import get_window
from typing import Optional, Tuple
import warnings
import random
import math
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass

EPOCHS = 15
BATCH_SIZE = 32
IMAGE_SIZE = [512, 512]
# Seed
SEED = 21
# Learning rate
LR = 0.0001
# Verbosity
VERBOSE = 2


# Function to create cqt kernel
def create_cqt_kernels(
		q: float,
		fs: float,
		fmin: float,
		n_bins: int = 84,
		bins_per_octave: int = 12,
		norm: float = 1,
		window: str = "hann",
		fmax: Optional[float] = None,
		topbin_check: bool = True
) -> Tuple[np.ndarray, int, np.ndarray, float]:
	fft_len = 2 ** _nextpow2(np.ceil(q * fs / fmin))

	if (fmax is not None) and (n_bins is None):
		n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
		freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
	elif (fmax is None) and (n_bins is not None):
		freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
	else:
		warnings.warn("If nmax is given, n_bins will be ignored", SyntaxWarning)
		n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
		freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

	if np.max(freqs) > fs / 2 and topbin_check:
		raise ValueError(f"The top bin {np.max(freqs)} Hz has exceeded the Nyquist frequency, \
                           please reduce the `n_bins`")

	kernel = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)

	length = np.ceil(q * fs / freqs)
	for k in range(0, int(n_bins)):
		freq = freqs[k]
		l = np.ceil(q * fs / freq)

		if l % 2 == 1:
			start = int(np.ceil(fft_len / 2.0 - l / 2.0)) - 1
		else:
			start = int(np.ceil(fft_len / 2.0 - l / 2.0))

		sig = get_window(window, int(l), fftbins=True) * np.exp(
			np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * freq / fs) / l

		if norm:
			kernel[k, start:start + int(l)] = sig / np.linalg.norm(sig, norm)
		else:
			kernel[k, start:start + int(l)] = sig
	return kernel, fft_len, length, freqs


def _nextpow2(a: float) -> int:
	return int(np.ceil(np.log2(a)))


# Function to prepare cqt kernel
def prepare_cqt_kernel(
		sr=22050,
		hop_length=512,
		fmin=32.70,
		fmax=None,
		n_bins=84,
		bins_per_octave=12,
		norm=1,
		filter_scale=1,
		window="hann"
):
	q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
	print(q)
	return create_cqt_kernels(q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)


# Function to create cqt image
def create_cqt_image(wave, hop_length=16):
	CQTs = []
	for i in range(3):
		x = wave[i]
		x = tf.expand_dims(tf.expand_dims(x, 0), 2)
		x = tf.pad(x, PADDING, "REFLECT")

		CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
		CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
		CQT_real *= tf.math.sqrt(LENGTHS)
		CQT_imag *= tf.math.sqrt(LENGTHS)

		CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
		CQTs.append(CQT[0])
	return tf.stack(CQTs, axis=2)


HOP_LENGTH = 6
cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
	sr=2048,
	hop_length=HOP_LENGTH,
	fmin=20,
	fmax=500,
	bins_per_octave=9)
LENGTHS = tf.constant(lengths, dtype=tf.float32)
CQT_KERNELS_REAL = tf.constant(np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, 2))
CQT_KERNELS_IMAG = tf.constant(np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, 2))
PADDING = tf.constant([[0, 0],
                       [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                       [0, 0]])

conv = tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same')
from tensorflow.keras import layers


class BasicConv(object):
	def __init__(self, out_planes, kernel_size):
		super(BasicConv, self).__init__()
		self.conv = tf.keras.layers.Conv2D(
			out_planes,
			kernel_size=[kernel_size, kernel_size],
			strides=[1, 1],

			use_bias=False,
			data_format='channels_first')
		self.bn = tf.keras.layers.BatchNormalization(
			axis=-1,
			momentum=0.999,
			epsilon=1e-5,
			fused=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = tf.nn.relu(x)
		return x


class ChannelPool(object):
	def forward(self, x):
		return tf.concat([tf.expand_dims(tf.reduce_max(x, axis=1), axis=1),
		                  tf.expand_dims(tf.reduce_mean(x, axis=1), axis=1)], axis=1)


class SpatialGate(object):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 12
		self.compress = ChannelPool()
		self.spatial = BasicConv(1, kernel_size)

	def forward(self, x):
		x_compress = self.compress.forward(x)
		x_out = self.spatial.forward(x_compress)
		scale = tf.nn.sigmoid(x_out)
		return x * scale


class TripletAttention(tf.keras.layers.Layer):
	def __init__(self, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
		super(TripletAttention, self).__init__()
		self.ChannelGateH = SpatialGate()
		self.ChannelGateW = SpatialGate()
		self.no_spatial = no_spatial
		if not no_spatial:
			self.SpatialGate = SpatialGate()

	def forward(self, x):
		x_perm1 = tf.transpose(x, perm=[0, 2, 1, 3])
		x_out1 = self.ChannelGateH.forward(x_perm1)
		x_out11 = tf.transpose(x_out1, perm=[0, 2, 1, 3])
		x_perm2 = tf.transpose(x, perm=[0, 3, 2, 1])
		x_out2 = self.ChannelGateW.forward(x_perm2)
		x_out21 = tf.transpose(x_out2, perm=[0, 3, 2, 1])
		if not self.no_spatial:
			x_out = self.SpatialGate.forward(x)
			x_out = (1 / 3) * (x_out + x_out11 + x_out21)
		else:
			x_out = (1 / 2) * (x_out11 + x_out21)
		return x_out


triplet = TripletAttention()


def prepare_image(wave):
	# Decode raw
	wave = tf.convert_to_tensor(np.load(r"F:\Pycharm_projects\Gnet2\data\00000e74ad.npy"))
	normalized_waves = []
	# Normalize
	for i in range(3):
		normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
		normalized_waves.append(normalized_wave)
	# Stack and cast
	wave = tf.stack(normalized_waves)
	wave = tf.cast(wave, tf.float32)
	# Create image
	image = create_cqt_image(wave, HOP_LENGTH)
	# Resize image

	image = tf.squeeze(triplet(image[tf.newaxis, :, :, :]))
	image = tf.image.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[0]))
	# Reshape
	image = tf.reshape(image, [*IMAGE_SIZE, 3])

	return image


import numpy as np


def generalized_mean_pool_2d(X, gm_exp):
	pool = (tf.reduce_mean(tf.abs(X ** (gm_exp)),
	                       axis=[1, 2],
	                       keepdims=False) + 1.e-8) ** (1. / gm_exp)
	return pool


def get_model():
	with strategy.scope():
		inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
		x = efn.EfficientNetB7(include_top=False, weights='noisy-student')(inp)
		x = GeMPoolingLayer(p=4)(x)
		output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
		model = tf.keras.models.Model(inputs=[inp], outputs=[output])
		opt = tf.keras.optimizers.Adam(learning_rate=LR)
		opt = tfa.optimizers.SWA(opt)
		model.compile(
			optimizer=opt,
			loss=[tf.keras.losses.BinaryCrossentropy()],
			metrics=[tf.keras.metrics.AUC()]
		)
		return model
