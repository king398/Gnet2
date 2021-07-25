# Asthetics
import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from kaggle_datasets import KaggleDatasets
from glob import glob
import pandas as pd
import numpy as np
import os
import time
import cv2
import random
import shutil
import math
import re

pd.set_option('display.max_columns', None)

# Visualizations
from PIL import Image
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
% matplotlib
inline
sns.set(style="whitegrid")

# Machine Learning
# Pre Procesing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Models
from sklearn.model_selection import train_test_split, KFold
# Deep Learning
import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score, FBetaScore
from tensorflow_addons.callbacks import TQDMProgressBar
from tensorflow.keras.utils import plot_model
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
# Metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

print('TF', tf.__version__)

# Random Seed Fixing
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)


seed_everything()

# Model Params
KFOLDS = 4
IMG_SIZES = [512]*KFOLDS
BATCH_SIZES = [2]*KFOLDS
EPOCHS = [15]*KFOLDS
EFF_NETS = [1]*KFOLDS # WHICH EFFICIENTNET B? TO USE

# Model Eval Params
DISPLAY_PLOT = True

# Inference Params
WGTS = [1/KFOLDS]*KFOLDS
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


strategy, TPU_DETECTED = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
from tqdm.notebook import tqdm

files_test_g = []
for i, k in tqdm([(0, 1), (2, 3), (4, 5), (6, 7)]):
	GCS_PATH = KaggleDatasets().get_gcs_path(f'cqt-g2net-test-{i}-{k}')
	files_test_g.extend(np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec'))).tolist())
num_train_files = len(files_test_g)
print(files_test_g[0])
print('test_files:', num_train_files)
print()
skf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_pred = [];
oof_tar = [];
oof_val = [];
oof_f1 = [];
oof_ids = [];
oof_folds = []

files_test_g = np.array(files_test_g)
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]


def build_model(size, ef=0):
	inp = tf.keras.layers.Input(shape=(size, size, 3))
	base = EFNS[ef](input_shape=(size, size, 3), weights='imagenet', include_top=False)

	x = base(inp)

	x = tf.keras.layers.GlobalAvgPool2D()(x)

	x = tf.keras.layers.Dropout(0.)(x)

	x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	model = tf.keras.Model(inputs=inp, outputs=x)
	lr_decayed_fn = tf.keras.experimental.CosineDecay(
		6e-4,
		820 * 1,
	)

	opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=7e-4)
	loss = tf.keras.losses.BinaryCrossentropy()
	model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
	return model
for fold in range(0, KFOLDS):

	print('#' * 25);
	print('#### FOLD', fold + 1)
	# BUILD MODEL
	K.clear_session()

	with strategy.scope():
		model = build_model(IMG_SIZES[fold], ef=EFF_NETS[fold])
	print('\tLoading model...')

	model.load_weights(f'../input/gnet-models/fold-{fold}.h5')

	print('\tPredict...')
	ds_test = get_dataset(files_test_g, labeled=False, return_image_ids=True,
	                      repeat=False, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 2)

	_oof_pred = []
	_oof_tar = []
	for img, target in tqdm(iter(ds_test)):
		p = model.predict_on_batch(img).flatten()

		t = target.numpy().flatten()
		_oof_pred.extend(p.tolist())
		_oof_tar.extend(t.tolist())
		K.clear_session()
	oof_pred.append(np.array(_oof_pred).flatten())
	oof_ids.append(np.array(_oof_tar).flatten())
	sns.distplot(oof_pred[-1])
	plt.show()
	print('\tFinished...')
sub = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')
sub['id'] = [t.decode("utf-8") for t in oof_ids[-1]]
sub['target'] = np.mean(oof_pred, axis=0)
sub = sub.sort_values('id')
sub.head()
sub.to_csv('submission.csv', index=False)
