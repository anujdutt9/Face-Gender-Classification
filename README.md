# Face Gender Classification

# Usage

1. Install Miniconda. Follow instructions here: https://docs.conda.io/en/latest/miniconda.html

2. Once ready, use the environment.yml file to get the virtual environment up and running using the following command:
```
    conda env create -f environment.yml
```

3. Download the pre-trained Caffe model from here: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/.
   
   Also download the fold_0_data.txt to fold_4_data.txt files for getting the labels.

4. Convert the Caffe model to ONNX model using this: https://github.com/htshinichi/caffe-onnx
```    
    python convert2onnx.py \
          vgg_face_caffe/VGG_FACE_deploy.prototxt \
          vgg_face_caffe/VGG_FACE.caffemodel \
          vgg_face onnxmodel
```

5. Convert the ONNX model to Keras Model using this: https://github.com/nerox8664/onnx2keras

   For conversion of onnx model to keras model, run the following jupyter notebook.
```
    ONNX to Keras.ipynb
```

6. Once you have the Keras model ready, we are now ready for Transfer Learning.

7. In the Model Training and Prediction directory, unzip the faces.zip file and run the following Jupyter Notebook:
```
    Face_gender_classification.ipynb
```

This above notebook does the following:

a) Loads in the Dataset using the fold_x_data.txt files.

b) Splits the data into Train/Val/Test set.

c) Converts and Saves the dataset into TFRecords format for efficiency.

d) Loads the keras model converted above.

e) Removes the unnecessary layers from the model.

f) Freezes the trained layers and adds the new layers on top of it.

g) Defines the Loss and optimizer functions.

h) Trains the model for set number of epochs by streaming data efficiently from TFRecords using tf.data API in batches.

i) Saves the trained model as well as it's chekpoints.

j) Shows a plot between model loss and accuracy w.r.t number of epochs.

k) Makes predictions on the test dataset and computes accuracy, Precision, Recall, F1 Score and the confusion matrix.

l) Shows an example of how to load the trained model for making inference on random images.

