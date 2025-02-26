# cat-and-dog-classification
Cats and Dogs Classification using TensorFlow/Keras
This repository contains code for a simple Convolutional Neural Network (CNN) to classify images of cats and dogs.

Dataset
The dataset used is the "Cats and Dogs Filtered" dataset, available on Kaggle. It contains a subset of images from the original "Cats vs. Dogs" dataset, pre-split into training and validation sets.

You can download the dataset using the following Kaggle Hub command within your notebook:
import kagglehub
path = kagglehub.dataset_download("birajsth/cats-and-dogs-filtered")
print("Path to dataset files:", path)

Cats and Dogs Classification using TensorFlow/Keras
This repository contains code for a simple Convolutional Neural Network (CNN) to classify images of cats and dogs.

Dataset
The dataset used is the "Cats and Dogs Filtered" dataset, available on Kaggle. It contains a subset of images from the original "Cats vs. Dogs" dataset, pre-split into training and validation sets.

You can download the dataset using the following Kaggle Hub command within your notebook:

Python

import kagglehub
path = kagglehub.dataset_download("birajsth/cats-and-dogs-filtered")
print("Path to dataset files:", path)
Dependencies
Python 3.x
TensorFlow/Keras
NumPy
Matplotlib
OpenCV (cv2)
Kaggle Hub
pydot (for model visualization)
You can install the necessary libraries using pip:

pip install tensorflow numpy matplotlib opencv-python kagglehub pydot

Code Description
The code is organized into a Jupyter Notebook (.ipynb file). It performs the following steps:

Import Libraries: Imports necessary libraries for building and training the CNN.
Download Dataset: Downloads the Cats and Dogs Filtered dataset using Kaggle Hub.
Data Loading and Preprocessing:
Specifies the paths to the training and validation directories.
Counts the number of cat and dog images in each directory.
Creates ImageDataGenerator instances for data augmentation and preprocessing (rescaling).
Loads the images using flow_from_directory.
Visualizes a sample of training images.
Model Building:
Builds a simple CNN model using Keras Sequential API.
The model consists of convolutional layers, max-pooling layers, a flatten layer, and dense layers.
The1 output layer uses a sigmoid activation function for binary classification.   

Model Compilation and Training:
Compiles the model with the Adam optimizer and binary cross-entropy loss.
Trains the model using the training data generator and validates it using the validation data generator.
Saves the trained model to 'model.h5'
Model Conversion to TensorFlow Graph:
Converts the trained Keras model into a frozen TensorFlow graph (.xml file). This is done to create a portable model that can be used in other applications.
Prints the input and output node names.
Training and Validation Visualization:
Plots the training and validation accuracy and loss curves.
Image Prediction:
Loads the saved model
Loads and preprocesses a test image ("dog.jpg").
Makes a prediction using the loaded model.
Prints "Dog" or "Cat" based on the prediction.

Usage
 Clone the repository:

Bash

git clone [repository URL]
cd [repository directory]
Install dependencies:

Bash

pip install -r requirements.txt
(If you create a requirements.txt file with the dependencies)

Run 

Bash

 Cats_Dogs_Classification.ipynb


Place a "dog.jpg" file in the same directory as the notebook to test the prediction.

Model Saving
The trained model is saved as model.h5. A frozen tensorflow graph is also saved as model.xml

Results
The  training and validation accuracy and loss curves, as well as the prediction result for the test image.

Further Improvements
Experiment with different model architectures and hyperparameters.
Implement data augmentation techniques to improve model generalization.
Evaluate the model on a larger test set.
Convert the saved model to TensorFlow Lite for mobile deployment.
Add dropout layers to reduce overfitting.
Increase the number of epochs.
Implement early stopping.
