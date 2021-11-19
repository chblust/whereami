import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from model_builder import build_datasets, build_model

import numpy as np

train_ds, val_ds = build_datasets()

model = build_model()
model.load_weights("model")
model.compile()

predictions = model.predict(val_ds)
predicted_indices = tf.argmax(predictions, 1)
labels = np.concatenate([y for x, y in val_ds], axis=0)
names = ""
for name in val_ds.class_names:
    names += name[0:3] + ' '
print('   ' + names)
conf = tf.math.confusion_matrix(labels, predicted_indices)

tp_tn = 0
tp_fn_fp_tn = 0
for i in range(0,len(val_ds.class_names)):
    tp_tn += conf[i][i]
    for j in range(0,len(val_ds.class_names)):
        tp_fn_fp_tn += conf[i][j]

acc = tp_tn / tp_fn_fp_tn
print(f"Accuracy: {acc*100:.03f}")