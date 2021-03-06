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
from tensorflow.keras import mixed_precision

NUM_FOLDS = 4
IMAGE_SIZE = 512
BATCH_SIZE = 12
EFFICIENTNET_SIZE = 7
WEIGHTS = "noisy-student"

MIXUP_PROB = 0.0
EPOCHS = 18
R_ANGLE = 0 / 180 * np.pi
S_SHIFT = 0.0
T_SHIFT = 0.0
LABEL_POSITIVE_SHIFT = 0.96
SAVEDIR = Path("models")
SAVEDIR.mkdir(exist_ok=True)

OOFDIR = Path("oof")
OOFDIR.mkdir(exist_ok=True)


def set_seed(seed=42):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


set_seed(1213)


def auto_select_accelerator():
	TPU_DETECTED = False
	try:
		tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
		tf.config.experimental_connect_to_cluster(tpu)
		tf.tpu.experimental.initialize_tpu_system(tpu)
		strategy = tf.distribute.experimental.TPUStrategy(tpu)
		print("Running on TPU:", tpu.master())
		policy = mixed_precision.Policy('mixed_bfloat16')
		mixed_precision.set_global_policy(policy)
		tf.config.optimizer.set_jit(True)
		TPU_DETECTED = True
	except ValueError:
		strategy = tf.distribute.get_strategy()
	print(f"Running on {strategy.num_replicas_in_sync} replicas")

	return strategy, TPU_DETECTED


strategy, tpu_detected = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = 8
# gcs_paths = []
# for i, j in [(0, 4), (5, 9), (10, 14), (15, 19)]:
#     GCS_path = KaggleDatasets().get_gcs_path(f"g2net-waveform-tfrecords-train-{i}-{j}")
#     gcs_paths.append(GCS_path)
#     print(GCS_path)


gcs_paths = [
	"gs://kds-68dcd9850ea61ccbd6899e8c2e96fdac52b1b7b779d4ca58caca272e",
	"gs://kds-3232b2fd9a72814c605032010853239f477f398aeae2448327ce82dd",
	"gs://kds-08b564b271d72b224e01986c7f62bb7bb5b595df73a10242a6881d6c",
	"gs://kds-c25102ee23417a87f8dfd5828737dc97c3be9c783200bef681528fe9"

]
all_files = []
for path in gcs_paths:
	all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/train*.tfrecords"))))
#     all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/train*.tfrec"))))

print("train_files: ", len(all_files))


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


HOP_LENGTH = 6
cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
	sr=2048,
	hop_length=HOP_LENGTH,
	fmin=20,
	fmax=1024,
	bins_per_octave=9)
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


def read_labeled_tfrecord(example):
	tfrec_format = {
		"wave": tf.io.FixedLenFeature([], tf.string),
		"wave_id": tf.io.FixedLenFeature([], tf.string),
		"target": tf.io.FixedLenFeature([], tf.int64)
	}
	example = tf.io.parse_single_example(example, tfrec_format)
	return prepare_image(example["wave"], IMAGE_SIZE), tf.reshape(tf.cast(example["target"], tf.float32), [1])


def read_unlabeled_tfrecord(example, return_image_id):
	tfrec_format = {
		"wave": tf.io.FixedLenFeature([], tf.string),
		"wave_id": tf.io.FixedLenFeature([], tf.string)
	}
	example = tf.io.parse_single_example(example, tfrec_format)
	return prepare_image(example["wave"], IMAGE_SIZE), example["wave_id"] if return_image_id else 0


def count_data_items(fileids):
	return len(fileids) * 28000


def count_data_items_test(fileids):
	return len(fileids) * 22600


# def count_data_items(fileids):
#     # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
#     n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in fileids]
#     return np.sum(n)


def mixup(image, label, probability=0.5, aug_batch=64 * 8):
	imgs = []
	labs = []
	for j in range(aug_batch):
		p = tf.cast(tf.random.uniform([], 0, 1) <= probability, tf.float32)
		k = tf.cast(tf.random.uniform([], 0, aug_batch), tf.int32)
		a = tf.random.uniform([], 0, 1) * p
		img1 = image[j]
		img2 = image[k]
		imgs.append((1 - a) * img1 + a * img2)
		lab1 = label[j]
		lab2 = label[k]
		labs.append((1 - a) * lab1 + a * lab2)
	image2 = tf.reshape(tf.stack(imgs), (aug_batch, IMAGE_SIZE, IMAGE_SIZE, 3))
	label2 = tf.reshape(tf.stack(labs), (aug_batch,))
	return image2, label2


def time_shift(img, shift=T_SHIFT):
	if shift > 0:
		T = IMAGE_SIZE
		P = tf.random.uniform([], 0, 1)
		SHIFT = tf.cast(T * P, tf.int32)
		return tf.concat([img[-SHIFT:], img[:-SHIFT]], axis=0)
	return img


def rotate(img, angle=R_ANGLE):
	if angle > 0:
		P = tf.random.uniform([], 0, 1)
		A = tf.cast(angle * P, tf.float32)
		return tfa.image.rotate(img, A)
	return img


def spector_shift(img, shift=S_SHIFT):
	if shift > 0:
		T = IMAGE_SIZE
		P = tf.random.uniform([], 0, 1)
		SHIFT = tf.cast(T * P, tf.int32)
		return tf.concat([img[:, -SHIFT:], img[:, :-SHIFT]], axis=1)
	return img


def img_aug_f(img):
	img = time_shift(img)
	img = spector_shift(img)
	# img = rotate(img)
	return img


def imgs_aug_f(imgs, batch_size):
	_imgs = []
	DIM = IMAGE_SIZE
	for j in range(batch_size):
		_imgs.append(img_aug_f(imgs[j]))
	return tf.reshape(tf.stack(_imgs), (batch_size, DIM, DIM, 3))


def label_positive_shift(labels):
	return labels * LABEL_POSITIVE_SHIFT


def aug_f(imgs, labels, batch_size):
	imgs, label = mixup(imgs, labels, MIXUP_PROB, batch_size)
	imgs = imgs_aug_f(imgs, batch_size)
	return imgs, label_positive_shift(label)


def prepare_image(wave, dim=256):
	wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
	scaling = tf.constant([1.5e-20, 1.5e-20, 0.5e-20], dtype=tf.float64)

	normalized_waves = []
	for i in range(3):
		#         normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
		normalized_wave = wave[i] / scaling[i]
		normalized_waves.append(normalized_wave)
	wave = tf.stack(normalized_waves)
	wave = tf.cast(wave, tf.float32)
	image = create_cqt_image(wave, HOP_LENGTH)
	image = tf.image.resize(image, size=(dim, dim))
	return tf.reshape(image, (dim, dim, 3))


def get_dataset(files, batch_size=16, repeat=False, shuffle=False, aug=True, labeled=True, return_image_ids=True):
	ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
	ds = ds.cache()

	if repeat:
		ds = ds.repeat()

	if shuffle:
		ds = ds.shuffle(1024 * 2)
		opt = tf.data.Options()
		opt.experimental_deterministic = False
		ds = ds.with_options(opt)

	if labeled:
		ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
	else:
		ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids), num_parallel_calls=AUTO)

	ds = ds.batch(batch_size * REPLICAS)
	if aug:
		ds = ds.map(lambda x, y: aug_f(x, y, batch_size * REPLICAS), num_parallel_calls=AUTO)
	ds = ds.prefetch(AUTO)
	return ds


from tensorflow import keras

from classification_models.tfkeras import Classifiers


def get_model(size=256, efficientnet_size=0, weights="noisy-student", count=0):
	with strategy.scope():
		inputs = tf.keras.layers.Input(shape=(size, size, 3))
		seresnet34, preprocess_input = Classifiers.get('seresnet34')

		efn_layer = seresnet34((size, size, 3), weights='imagenet', include_top=False)

		x = efn_layer(inputs)
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		#     x = tf.keras.layers.Dropout(0.2)(x)
		x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
		model = tf.keras.Model(inputs=inputs, outputs=x)

		lr_decayed_fn = tf.keras.experimental.CosineDecay(1e-3, count)
		opt = tf.keras.optimizers.Adam(learning_rate=LR)
		opt = tfa.optimizers.SWA(opt)
		loss = tf.keras.losses.BinaryCrossentropy()
		model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
		return model
