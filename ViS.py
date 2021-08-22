import os
import math
import random
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.signal import get_window
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

x = np.load(r"F:\Pycharm_projects\Gnet2\data\00000e74ad.npy")
print(x)
plt.plot(x[0])

plt.show()


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


HOP_LENGTH = 16
cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
	sr=2048,
	hop_length=HOP_LENGTH,
	fmin=20,
	fmax=1024,
	bins_per_octave=24)
LENGTHS = tf.constant(lengths, dtype=tf.float32)
CQT_KERNELS_REAL = tf.constant(np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, 2))
CQT_KERNELS_IMAG = tf.constant(np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, 2))
PADDING = tf.constant([[0, 0],
                       [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                       [0, 0]])


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


images = [r"F:\Pycharm_projects\Gnet2\data\0a101ba5db.npy", r"F:\Pycharm_projects\Gnet2\data\0b0006d146.npy",
          r"F:\Pycharm_projects\Gnet2\data\a01322c660.npy"]


def prepare_image(dim=512):
	for x in images:

		wave = np.load(x)
		normalized_waves = []
		for i in range(3):
			normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
			normalized_waves.append(normalized_wave)
		wave = tf.stack(normalized_waves)
		wave = tf.cast(wave, tf.float32)
		image = create_cqt_image(wave, HOP_LENGTH)
		image = tf.image.resize(image, size=(dim, dim))

		image = tfa.image.sharpness(image, factor=0.2)


		plt.imshow(image)

		plt.show()


prepare_image()
