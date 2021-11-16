import tensorflow as tf
from tensorflow.keras import datasets, layers, models

train_ds = tf.keras.utils.image_dataset_from_directory(
  'dataSet',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(360, 640),
  batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'dataSet',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(360, 640),
  batch_size=32)

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(360, 640, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds,epochs=10, validation_data=val_ds)