import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144
VAL_SPLIT = 0.2
DATASET_DIR = "dataSet"
BATCH_SIZE = 32

def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE)
    return train_ds, val_ds

def build_model():
    num_classes = 17

    model = tf.keras.Sequential([
    # tf.keras.layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    # tf.keras.layers.Conv2D(32, 3, activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(32, 3, activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    
    # tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten( input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes,activation='softmax')
    ])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'],)
            #   initializer=tf.keras.initializers.zeros)
    return model
