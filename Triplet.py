import os
import numpy as np
import pandas as pd
from scipy.signal import get_window
from typing import Optional, Tuple
import warnings
import random
import math
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass


class GeMPoolingLayer(tf.keras.layers.Layer):
	def __init__(self, p=1., train_p=False):
		super().__init__()
		if train_p:
			self.p = tf.Variable(p, dtype=tf.float32)
		else:
			self.p = p
		self.eps = 1e-6

	def call(self, inputs: tf.Tensor, **kwargs):
		inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
		inputs = tf.pow(inputs, self.p)
		inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
		inputs = tf.pow(inputs, 1. / self.p)
		return inputs


class BasicConv(tf.keras.Model):
	def __init__(self, out_planes, kernel_size, stride=1, padding="same", dilation=1, groups=1, relu=True,
	             bn=True, bias=False):
		super(BasicConv, self).__init__()

		self.out_channels = out_planes
		self.conv = tf.keras.layers.Conv2D(filters=out_planes, kernel_size=kernel_size, strides=stride, padding=padding,
		                                   dilation_rate=dilation, groups=groups, use_bias=bias)
		self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01, fused=True) if bn else None
		self.relu = tf.keras.layers.ReLU() if relu else None

	def call(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class ChannelPool(tf.keras.Model):
	def __init__(self):
		super(ChannelPool, self).__init__()

	def __call__(self, x):
		return tf.concat([tf.reduce_max(x, axis=[3], keepdims=True), tf.reduce_mean(x, axis=[3], keepdims=True)],
		                 axis=1)


class SpatialGate(tf.keras.Model):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 12
		self.compress = ChannelPool()
		self.spatial = BasicConv(out_planes=1, kernel_size=kernel_size, stride=1, relu=False)

	def __call__(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = tf.math.sigmoid(x_out)
		return x * scale


class TripletAttention(tf.keras.layers.Layer):
	def __init__(self, no_spatial=False):
		super(TripletAttention, self).__init__()
		self.ChannelGateH = SpatialGate()
		self.ChannelGateW = SpatialGate()
		self.no_spatial = no_spatial
		if not no_spatial:
			self.SpatialGate = SpatialGate()

	def call(self, x):
		x_perm1 = tf.transpose(x, perm=(0, 2, 1, 3))
		x_out1 = self.ChannelGateH(x_perm1)
		x_out11 = tf.transpose(x_out1, (0, 2, 1, 3))
		x_perm2 = tf.transpose(x, perm=(0, 3, 2, 1))
		x_out2 = self.ChannelGateW(x_perm2)
		x_out21 = tf.transpose(x_out2, (0, 3, 2, 1))
		if not self.no_spatial:
			x_out = self.SpatialGate(x)
			x_out = (1 / 3) * (x_out + x_out11 + x_out21)
		else:
			x_out = (1 / 2) * (x_out11 + x_out21)
		return x_out


model_aug = TripletAttention()


# Function to prepare image
def prepare_image(wave):
	# Decode raw
	wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
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
	image = tf.squeeze(triplet(tf.expand_dims(image, axis=1)))
	# Resize image
	image = tf.image.resize(image, [*IMAGE_SIZE])
	# Reshape
	image = tf.reshape(image, [*IMAGE_SIZE, 3])
	return tf.cast(image, tf.bfloat16)


def get_model():
	inp = tf.keras.layers.Input(shape=(512, 512, 3))

	x = efn.EfficientNetB7(include_top=False, weights='imagenet')(inp)
	x = tf.keras.layers.GlobalAveragePooling2D(p=4, train_p=False)(x)
	output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	model = tf.keras.models.Model(inputs=[inp], outputs=[output])
	opt = tf.keras.optimizers.Adam()
	opt = tfa.optimizers.SWA(opt)
	model.compile(
		optimizer=opt,
		loss=[tf.keras.losses.BinaryCrossentropy()],
		metrics=[tf.keras.metrics.AUC()]
	)
	return model


model = get_model()
