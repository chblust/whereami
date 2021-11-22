import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import model_builder
import sys, os

classes = ['amsterdam',  'auckland',  'new_york_city']
# Load the model
model = tf.keras.models.load_model('.')

if os.path.isdir(sys.argv[1]):
    paths = os.listdir(sys.argv[1])
    images = []
    index_counts = [0,0,0]
    for path in paths:
        image = Image.open(os.path.join(sys.argv[1], path))
        image = np.expand_dims(image, axis=0)
        # images.append(image)
        prediction = model.predict(image)[0]
        
        max_ind = np.argmax(prediction)
        index_counts[max_ind] += 1    
        # print(classes[max_ind])
    print(index_counts)
else:
    # train_ds, val_ds = model_builder.build_datasets()
    np_image = Image.open(sys.argv[1])
    # np_image = np.array(np_image).astype('float32')/255
    # np_image = transform.resize(np_image, (IMAGE_WIDTH, IMAG, 3))
    np_image = np.expand_dims(np_image, axis=0)

    # # run the inference
    prediction = model.predict(np_image)
    print(prediction)
