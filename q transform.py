import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers as L
import math
from tensorflow.keras.preprocessing import image
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import albumentations as A

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2, seed=42)
train = data.flow_from_directory(directory="", target_size=(256, 256), class_mode="sparse", subset="training")

valid = data.flow_from_directory(directory="", target_size=(256, 256), class_mode="sparse", subset="validation")

import efficientnet.tfkeras as efn

model = tf.keras.Sequential([L.InputLayer(input_shape=(69, 385, 1)), L.Conv2D(3, 3, activation='relu', padding='same'),
                             efn.EfficientNetB7(include_top=False, input_shape=(), weights='imagenet'),
                             L.GlobalAveragePooling2D(),
                             L.Dense(32, activation='relu'),
                             L.Dense(2, activation='sigmoid')])
best = tf.keras.callbacks.ModelCheckpoint("/content/Temp", monitor="val_auc", save_best_only=True,mode="max")
model.summary()


opt = tf.keras.optimizers.Adam(0.00005)
i = 0
print(len(train))

model.compile(optimizer=opt,
              loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
model.fit(train, epochs=5, validation_data=valid, callbacks=best)
