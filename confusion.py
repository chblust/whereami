import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import model_builder
import sys, os, threading

from string import ascii_uppercase
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt


cities = ['amsterdam', 'auckland', 'beijing', 'buenos_aires', 'cairo', 'cape_town', 'hong_kong', 'lagos', 'mexico_city', 'moscow', 'new_york_city', 'perth', 'rio', 'rome', 'santiago', 'sydney', 'vancouver']
# Load the model

conf_matrix =  t = [ [-1]*len(cities) for i in range(len(cities))]

conf_mutex = threading.Lock()
pred_mutex = threading.Lock()


def conf_thread(dir, row):
    model = tf.keras.models.load_model('.')
    paths = os.listdir(dir)
    images = []
    index_counts = [0]*len(cities)
    for i,path in enumerate(paths):
        image = Image.open(os.path.join(dir, path))
        image = np.expand_dims(image, axis=0)
        # images.append(image)
        # pred_mutex.acquire()
        prediction = model.predict(image)[0]
        # pred_mutex.release()
        max_ind = np.argmax(prediction)
        
        index_counts[max_ind] += 1   
        if i % 100 == 0:
            print(f"thread {dir} {i}/500") 
        
    print(f"Done with {dir}")
    conf_mutex.acquire()
    
    conf_matrix[row] = index_counts
    conf_mutex.release()



threads = []
for i, city in enumerate(cities):
    threads.append(threading.Thread(target=conf_thread, args=(f'dataSet/{city}', i)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

# conf_matrix = [[487, 13, 1], [10, 488, 2], [6, 4, 490]]
df_cm = DataFrame(conf_matrix, index=cities, columns=cities)

ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='g')
ax.set_xlabel('Predicted City')
ax.set_ylabel('Actual City')
plt.show()
print(conf_matrix)