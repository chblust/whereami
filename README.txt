WhereAmI is a neural network trained to identify cities from the given dataset by their streets.

The DataSet consists of a set of JPG images from each of 17 cities under the dataSet directory.

The model in stored in the tensorflow files, keras_metadata.pb and saved_model.pb.

To change the network architecture, edit model_builder.py

To retrain the network, run train.py

To evaluate the trained network, run eval.py

To generate a confusion matrix, run confusion.py

To calculate accuracy from confusion matrix output, paste conf. matrix output in acc.py and run it.

To generate a new dataset, use util/split_video.py