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


def mixup(image, label, PROBABILITY=1.0):
	# input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
	# output - a batch of images with mixup applied
	DIM = 128
	CLASSES = 2

	imgs = [];
	labs = []
	for j in range(256):
		# DO MIXUP WITH PROBABILITY DEFINED ABOVE
		P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
		# CHOOSE RANDOM
		k = tf.cast(tf.random.uniform([], 0, 256), tf.int32)
		a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
		# MAKE MIXUP IMAGE
		img1 = image[j,]
		img2 = image[k,]
		imgs.append((1 - a) * img1 + a * img2)
		# MAKE CUTMIX LABEL
		if len(label.shape) == 1:
			lab1 = tf.one_hot(label[j], CLASSES)
			lab2 = tf.one_hot(label[k], CLASSES)
		else:
			lab1 = label[j,]
			lab2 = label[k,]
		labs.append((1 - a) * lab1 + a * lab2)

	# RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
	image2 = tf.reshape(tf.stack(imgs), (256, 69, 385, 1))
	label2 = tf.reshape(tf.stack(labs), (256, CLASSES))
	return image2, label2


def id2path(idx, is_train=True):
	path = '../input/g2net-gravitational-wave-detection'
	if is_train:
		path = "/content/Train/" + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	else:
		path += '/test/' + idx[0] + '/' + idx[1] + '/' + idx[2] + '/' + idx + '.npy'
	return path


import torch
from nnAudio.Spectrogram import CQT1992v2


def increase_dimension(idx, is_train, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024,
                                                          hop_length=32)):
	images = []
	for i in idx:  # in order to use efficientnet we need 3 dimension images
		waves = np.load(id2path(i, is_train))
		waves = np.hstack(waves)
		waves = waves / np.max(waves)
		waves = torch.from_numpy(waves).float()
		image = transform(waves)
		image = np.array(image)
		image = np.transpose(image, (1, 2, 0))
		images.append(i)
	return images


example = np.load(path[4])
fig, a = plt.subplots(3, 1)
a[0].plot(example[1], color='green')
a[1].plot(example[1], color='red')
a[2].plot(example[1], color='yellow')
fig.suptitle('Target 1', fontsize=16)
plt.show()
plt.show()

train_idx = train_labels['id'].values
y = train_labels['target'].values
x_train, x_valid, y_train, y_valid = train_test_split(train_idx, y, test_size=0.2, random_state=42, stratify=y)
x_train = tf.constant(x_train)
y_train = tf.constant(y_train)
x_valid = tf.constant(x_valid)
y_valid = tf.constant(y_valid)
AUTO = tf.data.experimental.AUTOTUNE


def parse(images, labels, aug=aug()):
	images = np.array(increase_dimension(images, True))
	images = np.stack(images)
	images, labels = mixup(images, labels)
	images = tf.image.resize(images=images, size=(128, 128))

	return images, labels


train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train = train.map(map_func=parse, num_parallel_calls=AUTO).batch(batch_size=256)
for i in train:
	print(i)
	break
