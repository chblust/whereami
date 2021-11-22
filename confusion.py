import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import model_builder
from model_builder import build_datasets, build_model

import numpy as np

train_ds, val_ds = build_datasets()

model = models.load_model('.')

predictions = model.predict(train_ds, batch_size=model_builder.BATCH_SIZE)
predicted_indices = tf.argmax(predictions,1)
print(predicted_indices)
labels = np.concatenate([y for x, y in train_ds], axis=0)
# labels = np.flip(labels)
print(labels)
names = ""
for name in val_ds.class_names:
    names += name[0:3] + ' '
print('   ' + names)
conf = tf.math.confusion_matrix(labels, predicted_indices)
print(conf)
tp_tn = 0
tp_fn_fp_tn = 0
for i in range(0,len(val_ds.class_names)):
    tp_tn += conf[i][i]
    for j in range(0,len(val_ds.class_names)):
        tp_fn_fp_tn += conf[i][j]

acc = tp_tn / tp_fn_fp_tn
print(f"Accuracy: {acc*100:.03f}")