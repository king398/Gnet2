import os
import math
import random
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.signal import get_window
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(policy)
tf.__version__
IMAGE_SIZE = 256
BATCH_SIZE = 32
EFFICIENTNET_SIZE = 7
WEIGHTS = "noisy-student"


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
		TPU_DETECTED = True
	except ValueError:
		strategy = tf.distribute.get_strategy()
	print(f"Running on {strategy.num_replicas_in_sync} replicas")

	return strategy, TPU_DETECTED


strategy, tpu_detected = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync

gcs_paths = ['gs://kds-7b2c3c30216b2e0490017fad89bd4a82ba7b1b886619f56281b00bde',
             'gs://kds-7111a17939a58ba75017380c82cbec4bdbdaa330810e0acc24bdc8e8']

all_files = []
for path in gcs_paths:
	all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/test*.tfrecords"))))

print("test_files: ", len(all_files))
print(gcs_paths)


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


def prepare_image(wave, dim=256):
	wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
	normalized_waves = []
	for i in range(3):
		normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
		normalized_waves.append(normalized_wave)
	wave = tf.stack(normalized_waves)
	wave = tf.cast(wave, tf.float32)
	image = create_cqt_image(wave, HOP_LENGTH)
	image = tf.image.resize(image, size=(dim, dim))
	return tf.cast(tf.reshape(image, (dim, dim, 3)), dtype=tf.bfloat16)


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


def build_model(size=256, efficientnet_size=0, weights="imagenet", count=0):
	inputs = tf.keras.layers.Input(shape=(size, size, 3))

	efn_string = f"EfficientNetB{efficientnet_size}"
	efn_layer = getattr(efn, efn_string)(input_shape=(size, size, 3), weights=weights, include_top=False)

	x = efn_layer(inputs)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)

	x = tf.keras.layers.Dropout(0.2)(x)
	x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
	model = tf.keras.Model(inputs=inputs, outputs=x)

	lr_decayed_fn = tf.keras.experimental.CosineDecay(1e-3, count)
	opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=1e-4)
	loss = tf.keras.losses.BinaryCrossentropy()
	model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
	return model


files_test_all = np.array(all_files)
all_test_preds = []
with strategy.scope():
	model = build_model(
		size=IMAGE_SIZE,
		efficientnet_size=EFFICIENTNET_SIZE,
		weights=WEIGHTS,
		count=0)

weights_dir = Path("../input/noisy-student-models/models")
for i in range(4):
	print(f"Load weight for Fold {i + 1} model")
	model.load_weights(weights_dir / f"fold{i}.h5")

	ds_test = get_dataset(files_test_all, batch_size=BATCH_SIZE * 2, repeat=True, shuffle=False, aug=False,
	                      labeled=False, return_image_ids=False)
	STEPS = count_data_items_test(files_test_all) / BATCH_SIZE / 2 / REPLICAS
	pred = model.predict(ds_test, verbose=1, steps=STEPS)[:count_data_items_test(files_test_all)]
	all_test_preds.append(pred.reshape(-1))
ds_test = get_dataset(files_test_all, batch_size=BATCH_SIZE * 2, repeat=False, shuffle=False, aug=False, labeled=False,
                      return_image_ids=True)
file_ids = np.array([target.numpy() for img, target in iter(ds_test.unbatch())])
test_pred = np.zeros_like(all_test_preds[0])
for i in range(len(all_test_preds)):
	test_pred += all_test_preds[i] / len(all_test_preds)

test_df = pd.DataFrame({
	"id": [i.decode("UTF-8") for i in file_ids],
	"target": test_pred
})

test_df.head()
test_df.to_csv("submission.csv", index=False)
