# Asthetics
import warnings

import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import pandas as pd
import numpy as np
import random
import re

pd.set_option('display.max_columns', None)
import albumentations as A

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Machine Learning
# Pre Procesing
# Models
from sklearn.model_selection import KFold
# Deep Learning
import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import tensorflow_addons as tfa

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
# Metrics

print('TF', tf.__version__)

# Random Seed Fixing
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)


import os
from tensorflow.python.profiler import profiler_client
seed_everything()

tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', '8466')
print(profiler_client.monitor(tpu_profile_service_address, 100, 2))


# From https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
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


# Model Params
KFOLDS = 4
IMG_SIZES = [256] * KFOLDS
BATCH_SIZES = [42] * KFOLDS
EPOCHS = [30] * KFOLDS
EFF_NETS = [7] * KFOLDS  # WHICH EFFICIENTNET B? TO USE

AUG = True
MIX_UP_P = 0.1
S_SHIFT = 0.0
T_SHIFT = 0.0
R_ANGLE = 0 / 180 * np.pi

# Model Eval Params
DISPLAY_PLOT = True

# Inference Params
WGTS = [1 / KFOLDS] * KFOLDS
strategy, TPU_DETECTED = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
from tqdm.notebook import tqdm

files_train_g = []
for i, k in tqdm([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]):
	GCS_PATH = "gs://kds-45a91535658658322f84a0dd167c8ee2ed6d18989bfe3c7ea606e6ad"
	files_train_g.extend(np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'))).tolist())
num_train_files = len(files_train_g)
print(files_train_g)
print('train_files:', num_train_files)


def mixup(image, label, PROBABILITY=1.0, AUG_BATCH=BATCH_SIZES[0] * REPLICAS):
	# input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
	# output - a batch of images with mixup applied
	DIM = IMG_SIZES[0]

	imgs = [];
	labs = []
	for j in range(AUG_BATCH):
		# DO MIXUP WITH PROBABILITY DEFINED ABOVE
		P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
		# CHOOSE RANDOM
		k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
		a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
		# MAKE MIXUP IMAGE
		img1 = image[j,]
		img2 = image[k,]
		imgs.append((1 - a) * img1 + a * img2)
		# MAKE CUTMIX LABEL
		lab1 = label[j,]
		lab2 = label[k,]
		labs.append((1 - a) * lab1 + a * lab2)

	# RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
	image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
	label2 = tf.reshape(tf.stack(labs), (AUG_BATCH,))
	return image2, label2

def time_shift(img, shift=T_SHIFT):
	T = IMG_SIZES[0]
	P = tf.random.uniform([], 0, 1)
	SHIFT = tf.cast(T * P, tf.int32)
	return tf.concat([img[-SHIFT:], img[:-SHIFT]], axis=0)


def spector_shift(img, shift=S_SHIFT):
	T = IMG_SIZES[1]
	P = tf.random.uniform([], 0, 1)
	SHIFT = tf.cast(T * P, tf.int32)
	return tf.concat([img[:, -SHIFT:], img[:, :-SHIFT]], axis=1)


def rotate(img, angle=R_ANGLE):
	P = tf.random.uniform([], 0, 1)
	A = tf.cast(R_ANGLE * P, tf.float32)
	return tfa.image.rotate(img, A)


def img_aug_f(img):
	img = time_shift(img)
	img = spector_shift(img)
	img = rotate(img)
	return img


def imgs_aug_f(imgs, batch_size):
	_imgs = []
	DIM = IMG_SIZES[0]
	for j in range(batch_size):
		_imgs.append(img_aug_f(imgs[j]))
	return tf.reshape(tf.stack(_imgs), (batch_size, DIM, DIM, 3))


def aug_f(imgs, labels, batch_size):
	imgs, label = mixup(imgs, labels, MIX_UP_P, batch_size)
	imgs = imgs_aug_f(imgs, batch_size)
	return imgs, label


def read_labeled_tfrecord(example):
	tfrec_format = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'image_id': tf.io.FixedLenFeature([], tf.string),
		'target': tf.io.FixedLenFeature([], tf.int64)
	}
	example = tf.io.parse_single_example(example, tfrec_format)
	return prepare_image(example['image']), tf.reshape(tf.cast(example['target'], tf.float32), [1])


def read_unlabeled_tfrecord(example, return_image_id):
	tfrec_format = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'image_id': tf.io.FixedLenFeature([], tf.string),
	}
	example = tf.io.parse_single_example(example, tfrec_format)
	return prepare_image(example['image']), example['image_id'] if return_image_id else 0


def prepare_image(img, dim=IMG_SIZES[0]):
	img = tf.image.resize(tf.image.decode_png(img, channels=3), size=(dim, dim))
	img = tf.cast(img, tf.float32) / 255.0
	img = tf.reshape(img, [dim, dim, 3])

	return img


def count_data_items(fileids):
	n = [int(re.compile(r"-([0-9]*)\.").search(fileid).group(1))
	     for fileid in fileids]
	return np.sum(n)


def get_dataset(files, shuffle=False, repeat=False,
                labeled=True, return_image_ids=True, batch_size=16, dim=IMG_SIZES[0], aug=False):
	ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
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
		ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids),
		            num_parallel_calls=AUTO)

	ds = ds.batch(batch_size * REPLICAS)
	if aug:
		ds = ds.map(lambda x, y: aug_f(x, y, batch_size * REPLICAS), num_parallel_calls=AUTO)
	ds = ds.prefetch(AUTO)
	return ds


EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]


from classification_models.tfkeras import Classifiers


# for tensorflow.keras
# from classification_models.tfkeras import Classifiers


def build_model(size, ef=1, count=820):
	inp = tf.keras.layers.Input(shape=(size, size, 3))
	ResNet18, preprocess_input = Classifiers.get('resnext50') 
	base = ResNet18((256,256,3), weights='imagenet', include_top=False)

	x = base(inp)

	x = tf.keras.layers.Flatten()(x)

	x = tf.keras.layers.Dropout(0.2)(x)
	x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	model = tf.keras.Model(inputs=inp, outputs=x)
	lr_decayed_fn = tf.keras.experimental.CosineDecay(
		1e-3,
		count,
	)

	opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=1e-4)
	loss = tf.keras.losses.BinaryCrossentropy()
	model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
	return model


def vis_lr_callback(batch_size=8):
	lr_start = 1e-4
	lr_max = 0.000015 * REPLICAS * batch_size
	lr_min = 1e-5
	lr_ramp_ep = 4
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

	plt.figure(figsize=(10, 7))
	plt.plot([lrfn(i) for i in range(EPOCHS[0])])
	plt.show()


def get_lr_callback(batch_size=8):
	lr_start = 1e-4
	lr_max = 0.000015 * REPLICAS * batch_size
	lr_min = 1e-7
	lr_ramp_ep = 4
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

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
	return lr_callback


skf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_pred = [];
oof_tar = [];
oof_val = [];
oof_f1 = [];
oof_ids = [];
oof_folds = []

files_train_g = np.array(files_train_g)

for fold, (idxT, idxV) in enumerate(skf.split(files_train_g)):
	# CREATE TRAIN AND VALIDATION SUBSETS
	files_train = files_train_g[idxT]
	np.random.shuffle(files_train);
	files_valid = files_train_g[idxV]

	print('#' * 25);
	print('#### FOLD', fold + 1)

	train_images = count_data_items(files_train)
	val_images = count_data_items(files_valid)
	print('#### Training: %i | Validation: %i' % (train_images, val_images))

	# BUILD MODEL
	K.clear_session()
	with strategy.scope():
		model = build_model(IMG_SIZES[fold],
		                    count=count_data_items(files_train) / BATCH_SIZES[fold] // REPLICAS // 5)
	print('#' * 25)
	# SAVE BEST MODEL EACH FOLD
	sv = tf.keras.callbacks.ModelCheckpoint(
		'fold-%i.h5' % fold, monitor='val_auc', verbose=0, save_best_only=True,
		save_weights_only=True, mode='max', save_freq='epoch')

	# TRAIN
	print('Training...')
	history = model.fit(
		get_dataset(files_train, shuffle=True, repeat=True,
		            dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold], aug=AUG),
		epochs=EPOCHS[fold],
		callbacks=[sv, get_lr_callback(BATCH_SIZES[fold])],
		steps_per_epoch=count_data_items(files_train) / BATCH_SIZES[fold] // REPLICAS // 4,
		validation_data=get_dataset(files_valid, shuffle=False,
		                            repeat=False, dim=IMG_SIZES[fold]),
		verbose=1
	)

	# Loading best model for inference
	print('Loading best model...')
	model.load_weights('fold-%i.h5' % fold)

	ds_valid = get_dataset(files_valid, labeled=False, return_image_ids=False,
	                       repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 2)
	ct_valid = count_data_items(files_valid);
	STEPS = ct_valid / BATCH_SIZES[fold] / 2 / REPLICAS
	pred = model.predict(ds_valid, steps=STEPS, verbose=0)[:ct_valid, ]
	oof_pred.append(np.mean(pred.reshape((ct_valid, 1), order='F'), axis=1))

	# GET OOF TARGETS AND idS
	ds_valid = get_dataset(files_valid, repeat=False, dim=IMG_SIZES[fold],
	                       labeled=True, return_image_ids=True)
	oof_tar.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))

	# PLOT TRAINING
	if DISPLAY_PLOT:
		plt.figure(figsize=(8, 6))
		sns.distplot(oof_pred[-1])
		plt.show()

		plt.figure(figsize=(15, 5))
		plt.plot(np.arange(len(history.history['auc'])), history.history['auc'], '-o', label='Train auc',
		         color='#ff7f0e')
		plt.plot(np.arange(len(history.history['auc'])), history.history['val_auc'], '-o', label='Val auc',
		         color='#1f77b4')
		x = np.argmax(history.history['val_auc']);
		y = np.max(history.history['val_auc'])
		xdist = plt.xlim()[1] - plt.xlim()[0];
		ydist = plt.ylim()[1] - plt.ylim()[0]
		plt.scatter(x, y, s=200, color='#1f77b4');
		plt.text(x - 0.03 * xdist, y - 0.13 * ydist, 'max auc\n%.2f' % y, size=14)
		plt.ylabel('auc', size=14);
		plt.xlabel('Epoch', size=14)
		plt.legend(loc=2)
		plt2 = plt.gca().twinx()
		plt2.plot(np.arange(len(history.history['auc'])), history.history['loss'], '-o', label='Train Loss',
		          color='#2ca02c')
		plt2.plot(np.arange(len(history.history['auc'])), history.history['val_loss'], '-o', label='Val Loss',
		          color='#d62728')
		x = np.argmin(history.history['val_loss']);
		y = np.min(history.history['val_loss'])
		ydist = plt.ylim()[1] - plt.ylim()[0]
		plt.scatter(x, y, s=200, color='#d62728');
		plt.text(x - 0.03 * xdist, y + 0.05 * ydist, 'min loss', size=14)
		plt.ylabel('Loss', size=14)
		plt.title('FOLD %i - Image Size %i, %s' %
		          (fold + 1, IMG_SIZES[fold], EFNS[EFF_NETS[fold]].__name__), size=18)
		plt.legend(loc=3)
		plt.savefig(f'fig{fold}.png')
		plt.show()
