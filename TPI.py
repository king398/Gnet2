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
import albumentations as A

NUM_FOLDS = 4
IMAGE_SIZE = 256
BATCH_SIZE = 32
EFFICIENTNET_SIZE = 7
WEIGHTS = "imagenet"

MIXUP_PROB = 0.0
EPOCHS = 20
R_ANGLE = 0 / 180 * np.pi
S_SHIFT = 0.0
T_SHIFT = 0.0
LABEL_POSITIVE_SHIFT = 0.99
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
		TPU_DETECTED = True
	except ValueError:
		strategy = tf.distribute.get_strategy()
	print(f"Running on {strategy.num_replicas_in_sync} replicas")

	return strategy, TPU_DETECTED


strategy, tpu_detected = auto_select_accelerator()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
gcs_paths = ['gs://kds-43fefac8f343d9f8da0a5410d05d767d573979ec1ae47ce48623c21a',
             'gs://kds-bc9220faa6dec75bcf9f132b59a00137a7b8bf521231a78f9b2deb80',
             'gs://kds-20655d78c4f3ab25a78f671065d9f1d60a6219f31deab9367a1f19e7',
             'gs://kds-ecd5226e83bfa009993743d8203c1c1ed34bdcb09f1bc863c1998059']
all_files = []
for path in gcs_paths:
	all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/train*.tfrecords"))))

print("train_files: ", len(all_files))


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


aug = A.Compose([
	A.ToFloat(),
	A.augmentations.transforms.Sharpen(),
	A.augmentations.transforms.GaussNoise(),

])


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


def process_data(image):
	aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
	return aug_img


def aug_fn(image):
	data = {"image": image}
	aug_data = aug(**data)
	aug_img = aug_data["image"]
	aug_img = tf.cast(aug_img, tf.float32)
	return aug_img


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
	imgs = process_data(imgs)
	return imgs, label_positive_shift(label)


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


"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
	Activation,
	Add,
	BatchNormalization,
	Conv2D,
	Dense,
	DepthwiseConv2D,
	Dropout,
	GlobalAveragePooling2D,
	Input,
	PReLU,
	Reshape,
	Multiply,
)

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'

BLOCK_CONFIGS = {
	"b0": {  # width 1.0, depth 1.0
		"first_conv_filter": 32,
		"expands": [1, 4, 4, 4, 6, 6],
		"out_channels": [16, 32, 48, 96, 112, 192],
		"depthes": [1, 2, 2, 3, 5, 8],
		"strides": [1, 2, 2, 2, 1, 2],
		"use_ses": [0, 0, 0, 1, 1, 1],
	},
	"b1": {  # width 1.0, depth 1.1
		"first_conv_filter": 32,
		"expands": [1, 4, 4, 4, 6, 6],
		"out_channels": [16, 32, 48, 96, 112, 192],
		"depthes": [2, 3, 3, 4, 6, 9],
		"strides": [1, 2, 2, 2, 1, 2],
		"use_ses": [0, 0, 0, 1, 1, 1],
	},
	"b2": {  # width 1.1, depth 1.2
		"first_conv_filter": 32,
		"output_conv_filter": 1408,
		"expands": [1, 4, 4, 4, 6, 6],
		"out_channels": [16, 32, 56, 104, 120, 208],
		"depthes": [2, 3, 3, 4, 6, 10],
		"strides": [1, 2, 2, 2, 1, 2],
		"use_ses": [0, 0, 0, 1, 1, 1],
	},
	"b3": {  # width 1.2, depth 1.4
		"first_conv_filter": 40,
		"output_conv_filter": 1536,
		"expands": [1, 4, 4, 4, 6, 6],
		"out_channels": [16, 40, 56, 112, 136, 232],
		"depthes": [2, 3, 3, 5, 7, 12],
		"strides": [1, 2, 2, 2, 1, 2],
		"use_ses": [0, 0, 0, 1, 1, 1],
	},
	"s": {  # width 1.4, depth 1.8
		"first_conv_filter": 24,
		"output_conv_filter": 1280,
		"expands": [1, 4, 4, 4, 6, 6],
		"out_channels": [24, 48, 64, 128, 160, 256],
		"depthes": [2, 4, 4, 6, 9, 15],
		"strides": [1, 2, 2, 2, 1, 2],
		"use_ses": [0, 0, 0, 1, 1, 1],
	},
	"m": {  # width 1.6, depth 2.2
		"first_conv_filter": 24,
		"output_conv_filter": 1280,
		"expands": [1, 4, 4, 4, 6, 6, 6],
		"out_channels": [24, 48, 80, 160, 176, 304, 512],
		"depthes": [3, 5, 5, 7, 14, 18, 5],
		"strides": [1, 2, 2, 2, 1, 2, 1],
		"use_ses": [0, 0, 0, 1, 1, 1, 1],
	},
	"l": {  # width 2.0, depth 3.1
		"first_conv_filter": 32,
		"output_conv_filter": 1280,
		"expands": [1, 4, 4, 4, 6, 6, 6],
		"out_channels": [32, 64, 96, 192, 224, 384, 640],
		"depthes": [4, 7, 7, 10, 19, 25, 7],
		"strides": [1, 2, 2, 2, 1, 2, 1],
		"use_ses": [0, 0, 0, 1, 1, 1, 1],
	},
	"xl": {
		"first_conv_filter": 32,
		"output_conv_filter": 1280,
		"expands": [1, 4, 4, 4, 6, 6, 6],
		"out_channels": [32, 64, 96, 192, 256, 512, 640],
		"depthes": [4, 8, 8, 16, 24, 32, 8],
		"strides": [1, 2, 2, 2, 1, 2, 1],
		"use_ses": [0, 0, 0, 1, 1, 1, 1],
	},
}


def _make_divisible(v, divisor=4, min_value=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	"""
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
	return Conv2D(
		filters, kernel_size, strides=strides, padding=padding, use_bias=False,
		kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv"
	)(inputs)


def batchnorm_with_activation(inputs, activation="swish", name=""):
	"""Performs a batch normalization followed by an activation. """
	bn_axis = 1 if K.image_data_format() == "channels_first" else -1
	nn = BatchNormalization(
		axis=bn_axis,
		momentum=BATCH_NORM_DECAY,
		epsilon=BATCH_NORM_EPSILON,
		name=name + "bn",
	)(inputs)
	if activation:
		nn = Activation(activation=activation, name=name + activation)(nn)
	# nn = PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=name + "PReLU")(nn)
	return nn


def se_module(inputs, se_ratio=4, name=""):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

	filters = inputs.shape[channel_axis]
	# reduction = _make_divisible(filters // se_ratio, 8)
	reduction = filters // se_ratio
	# se = GlobalAveragePooling2D()(inputs)
	# se = Reshape((1, 1, filters))(se)
	se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
	se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER,
	            name=name + "1_conv")(se)
	# se = PReLU(shared_axes=[1, 2])(se)
	se = Activation("swish")(se)
	se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER,
	            name=name + "2_conv")(se)
	se = Activation("sigmoid")(se)
	return Multiply()([inputs, se])


def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, survival=None, use_se=0, is_fused=False, name=""):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	input_channel = inputs.shape[channel_axis]

	if is_fused and expand_ratio != 1:
		nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (3, 3), strides=stride, padding="same",
		                    name=name + "sortcut_")
		nn = batchnorm_with_activation(nn, name=name + "sortcut_")
	elif expand_ratio != 1:
		nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (1, 1), strides=(1, 1), padding="same",
		                    name=name + "sortcut_")
		nn = batchnorm_with_activation(nn, name=name + "sortcut_")
	else:
		nn = inputs

	if not is_fused:
		nn = DepthwiseConv2D(
			(3, 3), padding="same", strides=stride, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER,
			name=name + "MB_dw_"
		)(nn)
		nn = batchnorm_with_activation(nn, name=name + "MB_dw_")

	if use_se:
		nn = se_module(nn, se_ratio=4 * expand_ratio, name=name + "se_")

	# pw-linear
	if is_fused and expand_ratio == 1:
		nn = conv2d_no_bias(nn, output_channel, (3, 3), strides=stride, padding="same", name=name + "fu_")
		nn = batchnorm_with_activation(nn, name=name + "fu_")
	else:
		nn = conv2d_no_bias(nn, output_channel, (1, 1), strides=(1, 1), padding="same", name=name + "MB_pw_")
		nn = batchnorm_with_activation(nn, activation=None, name=name + "MB_pw_")

	if shortcut:
		if survival is not None and survival < 1:
			from tensorflow_addons.layers import StochasticDepth

			return StochasticDepth(float(survival))([inputs, nn])
		else:
			return Add()([inputs, nn])
	else:
		return nn


def EfficientNetV2(
		model_type,
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.2,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k-ft1k",
		name="EfficientNetV2",
):
	"""
	model_type: is the pre-defined model, value in ["s", "m", "l", "b0", "b1", "b2", "b3"].
	classes: Output classes number, 0 to exclude top layers.
	first_strides: is used in the first Conv2D layer.
	survivals: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
		Can be a constant value like `0.5` or `0.8`,
		or a tuple value like `(1, 0.8)` indicates the survival probability linearly changes from `1 --> 0.8` for `top --> bottom` layers.
		A higher value means a higher probability will keep the conv branch.
		or `None` to disable.
	pretrained: value in [None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]. Save path is `~/.keras/models/efficientnetv2/`.
	"""
	blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
	expands = blocks_config["expands"]
	out_channels = blocks_config["out_channels"]
	depthes = blocks_config["depthes"]
	strides = blocks_config["strides"]
	use_ses = blocks_config["use_ses"]
	first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
	output_conv_filter = blocks_config.get("output_conv_filter", 1280)

	inputs = Input(shape=input_shape)
	out_channel = _make_divisible(first_conv_filter, 8)
	nn = conv2d_no_bias(inputs, out_channel, (3, 3), strides=first_strides, padding="same", name="stem_")
	nn = batchnorm_with_activation(nn, name="stem_")

	# StochasticDepth survival_probability values
	total_layers = sum(depthes)
	if isinstance(survivals, float):
		survivals = [survivals] * total_layers
	elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
		start, end = survivals
		survivals = [start - (1 - end) * float(ii) / total_layers for ii in range(total_layers)]
	else:
		survivals = [None] * total_layers
	survivals = [survivals[int(sum(depthes[:id])): sum(depthes[: id + 1])] for id in range(len(depthes))]

	pre_out = out_channel
	for id, (expand, out_channel, depth, survival, stride, se) in enumerate(
			zip(expands, out_channels, depthes, survivals, strides, use_ses)):
		out = _make_divisible(out_channel, 8)
		is_fused = True if se == 0 else False
		for block_id in range(depth):
			stride = stride if block_id == 0 else 1
			shortcut = True if out == pre_out and stride == 1 else False
			name = "stack_{}_block{}_".format(id, block_id)
			nn = MBConv(nn, out, stride, expand, shortcut, survival[block_id], se, is_fused, name=name)
			pre_out = out

	output_conv_filter = _make_divisible(output_conv_filter, 8)
	nn = conv2d_no_bias(nn, output_conv_filter, (1, 1), strides=(1, 1), padding="valid", name="post_")
	nn = batchnorm_with_activation(nn, name="post_")

	if classes > 0:
		nn = GlobalAveragePooling2D(name="avg_pool")(nn)
		if dropout > 0 and dropout < 1:
			nn = Dropout(dropout)(nn)
		nn = Dense(classes, activation=classifier_activation, name="predictions")(nn)
	model = Model(inputs=inputs, outputs=nn, name=name)

	pretrained_dd = {"imagenet": "imagenet", "imagenet21k": "21k", "imagenet21k-ft1k": "21k-ft1k"}
	if pretrained in pretrained_dd:
		pre_url = "https://github.com/leondgarse/Keras_efficientnet_v2/releases/download/v1.0.0/efficientnetv2-{}-{}.h5"
		url = pre_url.format(model_type, pretrained_dd[pretrained])
		file_name = os.path.basename(url)
		pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models/efficientnetv2")
		model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
	return model


def EfficientNetV2B0(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.2,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2B0",
):
	return EfficientNetV2(model_type="b0", **locals())


def EfficientNetV2B1(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.2,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2B1",
):
	return EfficientNetV2(model_type="b1", **locals())


def EfficientNetV2B2(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.2,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2B2",
):
	return EfficientNetV2(model_type="b2", **locals())


def EfficientNetV2B3(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.2,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2B3",
):
	return EfficientNetV2(model_type="b3", **locals())


def EfficientNetV2S(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.2,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2S",
):
	return EfficientNetV2(model_type="s", **locals())


def EfficientNetV2M(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.3,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2M",
):
	return EfficientNetV2(model_type="m", **locals())


def EfficientNetV2L(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.4,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2L",
):
	return EfficientNetV2(model_type="l", **locals())


def EfficientNetV2XL(
		input_shape=(None, None, 3),
		classes=1000,
		dropout=0.4,
		first_strides=2,
		survivals=None,
		classifier_activation="softmax",
		pretrained="imagenet21k",
		name="EfficientNetV2XL",
):
	return EfficientNetV2(model_type="xl", **locals())


def get_actual_survival_probabilities(model):
	from tensorflow_addons.layers import StochasticDepth
	return [ii.survival_probability for ii in model.layers if isinstance(ii, StochasticDepth)]


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

def get_lr_callback(batch_size=8, replicas=8):
	lr_start = 1e-4
	lr_max = 0.000015 * replicas * batch_size
	lr_min = 1e-7
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

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
	return lr_callback


kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=1213)
oof_pred = []
oof_target = []

files_train_all = np.array(all_files)
for fold, (trn_idx, val_idx) in enumerate(kf.split(files_train_all)):
	if fold == 2:
		files_train = files_train_all[trn_idx]
		files_valid = files_train_all[val_idx]

		print("=" * 120)
		print(f"Fold {fold}")
		print("=" * 120)

		train_image_count = count_data_items(files_train)
		valid_image_count = count_data_items(files_valid)

		tf.keras.backend.clear_session()
		strategy, tpu_detected = auto_select_accelerator()
		with strategy.scope():
			model = build_model(
				size=IMAGE_SIZE,
				efficientnet_size=EFFICIENTNET_SIZE,
				weights=WEIGHTS,
				count=train_image_count // BATCH_SIZE // REPLICAS // 4)

		model_ckpt = tf.keras.callbacks.ModelCheckpoint(
			str(SAVEDIR / f"fold{fold}.h5"), monitor="val_auc", verbose=1, save_best_only=True,
			save_weights_only=True, mode="max", save_freq="epoch"
		)

		history = model.fit(
			get_dataset(files_train, batch_size=BATCH_SIZE, shuffle=True, repeat=True, aug=True),
			epochs=EPOCHS,
			callbacks=[model_ckpt, get_lr_callback(BATCH_SIZE, REPLICAS)],
			steps_per_epoch=train_image_count // BATCH_SIZE // REPLICAS // 4,
			validation_data=get_dataset(files_valid, batch_size=BATCH_SIZE * 4, repeat=False, shuffle=False, aug=False),
			verbose=1
		)

		print("Loading best model...")
		model.load_weights(str(SAVEDIR / f"fold{fold}.h5"))

		ds_valid = get_dataset(files_valid, labeled=False, return_image_ids=False, repeat=True, shuffle=False,
		                       batch_size=BATCH_SIZE * 2, aug=False)
		STEPS = valid_image_count / BATCH_SIZE / 2 / REPLICAS
		pred = model.predict(ds_valid, steps=STEPS, verbose=1)[:valid_image_count]
		oof_pred.append(np.mean(pred.reshape((valid_image_count, 1), order="F"), axis=1))

		ds_valid = get_dataset(files_valid, repeat=False, labeled=True, return_image_ids=True, aug=False)
		oof_target.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))

		plt.figure(figsize=(8, 6))
		sns.distplot(oof_pred[-1])
		plt.show()

		plt.figure(figsize=(15, 5))
		plt.plot(
			np.arange(len(history.history["auc"])),
			history.history["auc"],
			"-o",
			label="Train auc",
			color="#ff7f0e")
		plt.plot(
			np.arange(len(history.history["auc"])),
			history.history["val_auc"],
			"-o",
			label="Val auc",
			color="#1f77b4")

		x = np.argmax(history.history["val_auc"])
		y = np.max(history.history["val_auc"])

		xdist = plt.xlim()[1] - plt.xlim()[0]
		ydist = plt.ylim()[1] - plt.ylim()[0]

		plt.scatter(x, y, s=200, color="#1f77b4")
		plt.text(x - 0.03 * xdist, y - 0.13 * ydist, f"max auc\n{y}", size=14)

		plt.ylabel("auc", size=14)
		plt.xlabel("Epoch", size=14)
		plt.legend(loc=2)

		plt2 = plt.gca().twinx()
		plt2.plot(
			np.arange(len(history.history["auc"])),
			history.history["loss"],
			"-o",
			label="Train Loss",
			color="#2ca02c")
		plt2.plot(
			np.arange(len(history.history["auc"])),
			history.history["val_loss"],
			"-o",
			label="Val Loss",
			color="#d62728")

		x = np.argmin(history.history["val_loss"])
		y = np.min(history.history["val_loss"])

		ydist = plt.ylim()[1] - plt.ylim()[0]

		plt.scatter(x, y, s=200, color="#d62728")
		plt.text(x - 0.03 * xdist, y + 0.05 * ydist, "min loss", size=14)

		plt.ylabel("Loss", size=14)
		plt.title(f"Fold {fold + 1} - Image Size {IMAGE_SIZE}, EfficientNetB{EFFICIENTNET_SIZE}", size=18)

		plt.legend(loc=3)
		plt.savefig(OOFDIR / f"fig{fold}.png")
		plt.show()
