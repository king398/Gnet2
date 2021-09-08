import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
from CWT.cwt import ComplexMorletCWT

import warnings

image_size = [256, 256]

bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)
cwt_transform = ComplexMorletCWT(wavelet_width=8, fs=2048, lower_freq=20, upper_freq=500, n_scales=image_size[0],
                                 stride=int(np.ceil(4096 / image_size[0])), output='magnitude',
                                 data_format='channels_first')
warnings.filterwarnings("ignore")
image_size = [256, 256]
cwts_f = []
waves_f = []
num_samples = 4
bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)
cwt_transform = ComplexMorletCWT(wavelet_width=8, fs=2048, lower_freq=20, upper_freq=500, n_scales=image_size[0],
                                 stride=int(np.ceil(4096 / image_size[0])), output='magnitude',
                                 data_format='channels_first')
wavess = np.load(r"../input/g2net-gravitational-wave-detection/test/1/1/4/1141e7f00f.npy")

waves_f = []
cwts_f = []  # in order to use efficientnet we need 3 dimension images
for i in range(3):
	wave = wavess[i]  # for demonstration we will use only signal from one detector
	wave *= 1.3e+22  # normalization

	# With a filter
	wave_f = wave * signal.tukey(4096, 0.2)
	wave_f = signal.filtfilt(bHP, aHP, wave_f)
	waves_f.append(wave_f)
	wave_t_f = tf.convert_to_tensor(wave_f[np.newaxis, np.newaxis, :])
	cwt_f = cwt_transform(wave_t_f)
	cwts_f.append(np.squeeze(cwt_f.numpy()))
x = np.stack(cwts_f)
print(x.shape)
x = np.transpose(x)
normalized_waves = []
x = np.load("../input/g2net-gravitational-wave-detection/train/0/0/0/00001f4945.npy")
# for demonstration we will use only signal from one detector
cwts_f = []

for i in x:
	normalized_waves = x

	normalized_waves *= 1.3e+22  # normalization

	normalized_waves = signal.filtfilt(bHP, aHP, normalized_waves)
	normalized_waves = np.expand_dims(normalized_waves, axis=0)
	normalized_waves = tf.convert_to_tensor(normalized_waves)
	normalized_waves = cwt_transform(normalized_waves)
	cwts_f = []
	cwts_f.append(np.squeeze(normalized_waves.numpy()))
cwts_f = np.hstack(cwts_f)
