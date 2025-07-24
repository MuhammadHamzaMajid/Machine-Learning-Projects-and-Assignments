import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#set the random seed
tf.random.set_seed(42)

#1. PREPROCESS DATA(normalization)
valid_datagen = ImageDataGenerator(rescale = 1./255)

#this data is used in an augmented form
train_datagen_augmented = ImageDataGenerator(rescale = 1./255,
                                             rotation_range = 20,
                                             shear_range = 0.2,
                                             zoom_range = 0.2,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2,
                                             horizontal_flip = True)

#Setup paths to the directories
train_dir = "e:\\Deep Learning\\xrays\\train"
test_dir = "e:\\Deep Learning\\xrays\\test"

#2. IMPORT DATA FROM DIRECTORIES AND TURN IT INTO BATCHES
train_data_augmented = train_datagen_augmented.flow_from_directory(directory = train_dir,
                                               batch_size = 32,
                                               target_size = (224, 224),
                                               class_mode = "binary",
                                               seed = 42)

valid_data = valid_datagen.flow_from_directory(directory = test_dir,
                                               batch_size = 32,
                                               target_size = (224, 224),
                                               class_mode = "binary",
                                               seed = 42)

model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(filters = 10,
                           kernel_size = 3,
                           activation = "relu",
                           input_shape = (224, 224, 3)),

    tf.keras.layers.Conv2D(15, 3, activation = "relu"),
    tf.keras.layers.MaxPool2D(pool_size = 2, padding = "valid"),

    tf.keras.layers.Conv2D(15,3,activation = "relu"),

    tf.keras.layers.Conv2D(15, 3, activation = "relu"),

    tf.keras.layers.Conv2D(15, 3, activation = "relu"),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Conv2D(15, 3, activation = "relu"),
    tf.keras.layers.MaxPool2D(2),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = "sigmoid"),
    tf.keras.layers.Dense(256, activation = "sigmoid"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                metrics = ["accuracy"])

history_1 = model.fit(train_data_augmented,
                        validation_data = valid_data,
                        epochs = 10,
                        callbacks = [
                            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)  # Prevent overfitting
                        ])

import pandas as pd
import matplotlib.pyplot as plt
history_df_1 = pd.DataFrame(history_1.history)
history_df_1.loc[:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()
plt.show()