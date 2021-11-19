import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144
VAL_SPLIT = 0.2
DATASET_DIR = "simpleDataSet"

def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=32)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=32)
    return train_ds, val_ds

def inception_module(input, filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj):
    """
    Adapted from
    https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
    A blog by Faizan Shaikh.
    """    
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(input)
    
    conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(input)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(input)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    pool_proj = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(input)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    return layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

def build_model():    
    model = models.Sequential()
    # input = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    # model.add(input)
    model.add(layers.Conv2D(IMAGE_HEIGHT, (3,3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(IMAGE_HEIGHT*2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(IMAGE_HEIGHT*2, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # x = inception_module(input, filters_1x1=64,
    #                  filters_3x3_reduce=96,
    #                  filters_3x3=128,
    #                  filters_5x5_reduce=16,
    #                  filters_5x5=32,
    #                  filters_pool_proj=32)
    
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dense(18)(x)

    # model = models.Model(input, x)

    model.add(layers.Flatten())
    model.add(layers.Dense(IMAGE_HEIGHT*2, activation='relu'))
    model.add(layers.Dense(3))
    model.summary()
    return model