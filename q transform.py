import tensorflow as tf
import tensorflow.keras.layers as L

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train = data.flow_from_directory(directory="/content/Train/Train", target_size=(128, 128), class_mode="sparse",
                                 subset="training", batch_size=64)

valid = data.flow_from_directory(directory="/content/Train/Train", target_size=(128, 128), class_mode="sparse",
                                 subset="validation", batch_size=64)

import efficientnet.tfkeras as efn

model = tf.keras.Sequential([
	L.InputLayer(input_shape=(256, 256, 3)),
	efn.EfficientNetB7(include_top=False, input_shape=(), weights='imagenet'),
	L.GlobalAveragePooling2D(),
	L.Dense(32, activation='relu'),
	L.Dense(1, activation='sigmoid')])
best = tf.keras.callbacks.ModelCheckpoint("/content/Temp", monitor="val_auc", save_best_only=True, mode="max")
model.summary()

opt = tf.keras.optimizers.Adam(0.00005)
print(len(train))

model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.AUC()])
model.fit(train, epochs=5, validation_data=valid, callbacks=best)
