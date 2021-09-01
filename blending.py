import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore") 


sub1 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\submission.866XLcsv.csv').sort_values('id') 
sub2 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\submission869.csv').sort_values('id') 
preds1 = sub1.target
preds2 = sub2.target
sub = sub1.copy()
sub.loc[:, 'target'] =preds1*0.49+preds2*0.51
sub.to_csv('F:\Pycharm_projects\Gnet2\Sub/submission.csv', index=False)
    def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=64):
     """Build a 3D convolutional neural network model."""

     inputs = tf.keras.Input((width, height, depth, 1))
        x = layers.Conv3D(3, (3,3,3), strides=(1, 1, 1), 
                      padding='same', use_bias=True)(inputs)
    
    
    
    
     x = efn.EfficientNetB0(input_shape=(128, 128, 32, 3), weights='imagenet')(x)

     x = layers.GlobalAveragePooling3D()(x)
     x = layers.Dense(units=1024, activation="relu")(x)
     x = layers.Dropout(0.08)(x)

     outputs = layers.Dense(units=1, activation="sigmoid")(x)

	# Define the model.
     model = tf.keras.Model(inputs, outputs, name="3dcnn")

     return model

    with strategy.scope():
        model = get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=IMAGE_DEPTH)
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy", "AUC"])

    model.summary()̥

