!pip
install
efficientnet
tensorflow_addons > / dev / null
!pip
install - q
git + https: // github.com // Kevin - McIsaac / cmorlet - tensorflow @ Performance - -no - deps
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


def get_hardware_strategy():
	try:
		# TPU detection. No parameters necessary if TPU_NAME environment variable is
		# set: this is always the case on Kaggle.
		tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
		print('Running on TPU ', tpu.master())
	except ValueError:
		tpu = None

	if tpu:
		tf.config.experimental_connect_to_cluster(tpu)
		tf.tpu.experimental.initialize_tpu_system(tpu)
		strategy = tf.distribute.experimental.TPUStrategy(tpu)
		tf.config.optimizer.set_jit(True)
	else:
		# Default distribution strategy in Tensorflow. Works on CPU and single GPU.
		strategy = tf.distribute.get_strategy()

	print("REPLICAS: ", strategy.num_replicas_in_sync)
	return tpu, strategy


tpu, strategy = get_hardware_strategy()
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access (Train tf records)
GCS_PATH1 = "gs://kds-be973b4d24ed59842deeff7fc17c65f940acfe025d5bb2eb03261137"
GCS_PATH2 = "gs://kds-2a81609a438ac9f10d51ab54625a7f21f167acc05702fed3bfa405ab"
GCS_PATH3 = "gs://kds-9f5ea0ead6e9f5958187b731b5132e52b919b3173b744a0dc7c0c93c"
# Data access (Test tf records)
GCS_PATH4 = "gs://kds-5bd625816de42dc656f7ef6832ceedc0fd66cea6f4b639bf1489f4f0"
GCS_PATH5 = "gs://kds-42a3dac2586f0971b2a3b30162eed7cea42f2f9c34cbe66bd6c46b23"
print(GCS_PATH1, GCS_PATH2, GCS_PATH3, GCS_PATH4, GCS_PATH5)
# Configuration
EPOCHS = 20
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
# Seed
SEED = 1991
# Learning rate
LR = 0.0001
# Verbosity
VERBOSE = 1

# Training filenames directory
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH1 + '/train*.tfrec') + tf.io.gfile.glob(
	GCS_PATH2 + '/train*.tfrec') + tf.io.gfile.glob(GCS_PATH3 + '/train*.tfrec')
# Testing filenames directory
TESTING_FILENAMES = tf.io.gfile.glob(GCS_PATH4 + '/test*.tfrec') + tf.io.gfile.glob(GCS_PATH5 + '/test*.tfrec')


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
from CWT.cwt import ComplexMorletCWT

cwt_transform = ComplexMorletCWT(wavelet_width=8, fs=2048, lower_freq=20, upper_freq=500, n_scales=IMAGE_SIZE[0],
                                 stride=int(np.ceil(4096 / IMAGE_SIZE[0])), output='magnitude',
                                 data_format='channels_first')


def create_cqt_image(wave, hop_length=16):
	CQTs = []

	CQT = cwt_transform(tf.expand_dims(wave, axis=0))
	CQTs.append(CQT)
	return tf.convert_to_tensor(CQTs)


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


# Function to seed everything
def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	tf.random.set_seed(seed)


# Function to prepare image
def prepare_image(wave, dim=512):
	wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
	normalized_waves = []
	for i in range(3):
		normalized_wave = wave[i] / tf.math.reduce_max(wave[i])

		normalized_waves.append(normalized_wave)
	wave = tf.stack(normalized_waves)
	wave = tf.cast(wave, tf.float32)
	image = create_cqt_image(wave, HOP_LENGTH)
	image = tf.transpose(image[0, 0, :, :, :])
	image = tf.image.resize(image, size=(dim, dim))
	return tf.reshape(image, (dim, dim, 3))


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example):
	LABELED_TFREC_FORMAT = {
		'wave': tf.io.FixedLenFeature([], tf.string),
		'wave_id': tf.io.FixedLenFeature([], tf.string),
		'target': tf.io.FixedLenFeature([], tf.int64)
	}
	example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
	image = prepare_image(example['wave'])
	image_id = example['wave_id']
	target = tf.cast(example['target'], tf.float32)
	return image, image_id, target


# This function parse our images and also get the target variable
def read_unlabeled_tfrecord(example):
	LABELED_TFREC_FORMAT = {
		'wave': tf.io.FixedLenFeature([], tf.string),
		'wave_id': tf.io.FixedLenFeature([], tf.string)
	}
	example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
	image = prepare_image(example['wave'])
	image_id = example['wave_id']
	return image, image_id


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered=False, labeled=True):
	ignore_order = tf.data.Options()
	if not ordered:
		ignore_order.experimental_deterministic = False

	dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
	dataset = dataset.with_options(ignore_order)
	dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
	return dataset


# This function is to get our training dataset
def get_training_dataset(filenames, ordered=False, labeled=True):
	dataset = load_dataset(filenames, ordered=ordered, labeled=labeled)
	dataset = dataset.repeat()
	dataset = dataset.shuffle(2048)
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(AUTO)
	return dataset


# This function is to get our validation and test dataset
def get_val_test_dataset(filenames, ordered=True, labeled=True):
	dataset = load_dataset(filenames, ordered=ordered, labeled=labeled)
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(AUTO)
	return dataset


# Function to count how many photos we have in
def count_data_items(filenames):
	# The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
	n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
	return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TESTING_IMAGES = count_data_items(TESTING_FILENAMES)
print(f'Dataset: {NUM_TRAINING_IMAGES} training images')
print(f'Dataset: {NUM_TESTING_IMAGES} testing images')


def get_lr_callback():
	lr_start = 0.0001
	lr_max = 0.000015 * BATCH_SIZE
	lr_min = 0.0000001
	lr_ramp_ep = 3
	lr_sus_ep = 0
	lr_decay = 0.7

	def lrfn(epoch):
		if epoch < lr_ramp_ep:
			lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
		elif epoch < lr_ramp_ep + lr_sus_ep:
			lr = lr_max
		else:
			lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
		return lr

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=VERBOSE)
	return lr_callback


# Function to create our EfficientNetB7 model
def get_model():
	with strategy.scope():
		inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
		x = efn.EfficientNetB7(include_top=False, weights='imagenet')(inp)
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
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


# Function to train a model with 100% of the data
def train_and_evaluate():
	print('\n')
	print('-' * 50)
	print(f'Training EFFB7 with 100% of the data with seed {SEED} for {EPOCHS} epochs')
	if tpu:
		tf.tpu.experimental.initialize_tpu_system(tpu)
	train_dataset = get_training_dataset(TRAINING_FILENAMES, ordered=False, labeled=True)
	train_dataset = train_dataset.map(lambda image, image_id, target: (image, target))
	STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // (BATCH_SIZE * 4)
	K.clear_session()
	# Seed everything
	seed_everything(SEED)
	model = get_model()
	history = model.fit(train_dataset,
	                    steps_per_epoch=STEPS_PER_EPOCH,
	                    epochs=EPOCHS,
	                    callbacks=[get_lr_callback()],
	                    verbose=VERBOSE)

	print('\n')
	print('-' * 50)
	print('Test inference...')
	# Predict the test set
	dataset = get_val_test_dataset(TESTING_FILENAMES, ordered=True, labeled=False)
	image = dataset.map(lambda image, image_id: image)
	test_predictions = model.predict(image).astype(np.float32).reshape(-1)
	# Get the test set image_id
	image_id = dataset.map(lambda image, image_id: image_id).unbatch()
	image_id = next(iter(image_id.batch(NUM_TESTING_IMAGES))).numpy().astype('U')
	# Create dataframe output
	test_df = pd.DataFrame({'id': image_id, 'target': test_predictions})
	# Save test dataframe to disk
	test_df.to_csv(f'TEST_EfficientNetB7_{IMAGE_SIZE[0]}_{SEED}.csv', index=False)


train_and_evaluate()
