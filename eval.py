import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import model_builder

model = models.load_model('.')
# model.load_weights('model')
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['sparse_categorical_accuracy', 'accuracy'])
train_ds, val_ds = model_builder.build_datasets()

stats= model.evaluate(val_ds)
print(f"Accuracy: {stats[1]:.03f}")