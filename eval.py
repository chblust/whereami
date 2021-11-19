import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import model_builder

model = model_builder.build_model()
model.load_weights('model')
model.compile(metrics=['binary_accuracy', 'categorical_accuracy'])

train_ds, val_ds = model_builder.build_datasets()

stats= model.evaluate(val_ds)
print(f"Accuracy: {stats[1]:.03f}")