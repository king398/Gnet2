import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import math
from tensorflow.keras.preprocessing import image
from random import shuffle
from tensorflow.keras import mixed_precision
import sklearn.model_selection
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import efficientnet.tfkeras as efn

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

Datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train = Datagen.flow_from_directory(directory=r"/content/Train",
                                    class_mode="binary", shuffle=True,
                                    target_size=(256, 256), batch_size=256, subset="training")
validation = Datagen.flow_from_directory(directory=r"/content/Train",
                                         class_mode="binary", shuffle=True, batch_size=256)
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]


def build_model(size, ef=1, count=820):
	inp = tf.keras.layers.Input(shape=(size, size, 3))

	base = efn.EfficientNetB1(input_shape=(256, 256, 3), include_top=False
	                          )

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


model = build_model(size=256)

best = tf.keras.callbacks.ModelCheckpoint(filepath="/content/Data/Temp", save_best_only=True,
                                          monitor='val_auc', mode="max")
model.fit(train, validation_data=validation, epochs=5, callbacks=[best])
