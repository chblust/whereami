import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from model_builder import build_datasets, build_model
import numpy as np

train_ds, val_ds = build_datasets()

model = build_model()

model.fit(train_ds,epochs=1, validation_data=val_ds)

model.save('.')