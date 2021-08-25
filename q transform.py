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
from typing import Optional, Tuple
from scipy.signal import get_window
import warnings
train_labels = pd.read_csv('/content/Train/ing_labels.csv')
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

path = list(train_labels['id'])
for i in range(len(path)):
	path[i] = '/content/Train/' + path[i][0] + '/' + path[i][1] + '/' + path[i][2] + '/' + path[i] + '.npy'


def aug():
	return A.Compose([
		A.ToFloat(),
		A.augmentations.crops.transforms.RandomResizedCrop(height=128, width=128,
		                                                   always_apply=True, p=1.0),

	])


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
	wave = tf.random.shuffle(wave)
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


def prepare_image(wave, dim=128):
	wave = np.load(id2path(wave, True))
	normalized_waves = []
	for i in range(3):
		normalized_wave = wave[i] / tf.math.reduce_max(wave + [i])
		normalized_waves.append(normalized_wave)
	wave = tf.stack(normalized_waves)
	wave = tf.cast(wave, tf.float32)
	image = create_cqt_image(wave, HOP_LENGTH)
	image = tf.image.resize(image, size=(256, 128))

	return tf.reshape(image, (dim, dim, 3))


def id2path(idx, is_train=True):
	path = '../input/g2net-gravitational-wave-detection'
	if is_train:
		path = "/content/Train/" + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	else:
		path += '/test/' + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	return path




example = np.load(path[4])
fig, a = plt.subplots(3, 1)
a[0].plot(example[1], color='green')
a[1].plot(example[1], color='red')
a[2].plot(example[1], color='yellow')
fig.suptitle('Target 1', fontsize=16)
plt.show()
plt.show()


class Dataset(Sequence):
	def __init__(self, idx, y=None, batch_size=48, shuffle=True, valid=False, aug=aug()):
		self.idx = idx
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.valid = valid
		self.aug = aug
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

		batch_X = []
		for i in batch_ids:
			x = prepare_image(wave=i)
			batch_X.append(x)
		print(batch_X)

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
x_train, x_valid, y_train, y_valid = train_test_split(train_idx, y, test_size=0.05, random_state=42, stratify=y)
train_dataset = Dataset(x_train, y_train)
valid_dataset = Dataset(x_valid, y_valid, valid=True)
import efficientnet.tfkeras as efn

model = tf.keras.Sequential([
	efn.EfficientNetB7(include_top=False, input_shape=(), weights='imagenet'),
	L.GlobalAveragePooling2D(),
	L.Dense(32, activation='relu'),
	L.Dense(1, activation='sigmoid')])
best = tf.keras.callbacks.ModelCheckpoint("/content/Temp", monitor="val_auc", save_best_only=True)
model.summary()
lr_decayed_fn = tf.keras.experimental.CosineDecay(
	1e-3,
	700,
)

opt = tf.keras.optimizers.Adam(0.00005)
i = 0
print(len(train_dataset))

model.compile(optimizer=opt,
              loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
model.fit(train_dataset, epochs=5, validation_data=valid_dataset, callbacks=best)
