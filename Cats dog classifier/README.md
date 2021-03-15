# Dogs Cats Classifier

<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="Pandas" src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

# Aim

To build a deep learning model to classifiy dogs and cats' images.

## Feature

It can images of dogs and cats.

# Main Concept

### Convolutional Neural Network

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

# Dataset

For this project, The dataset was downloaded from kaggle.
Here's the link: https://www.kaggle.com/c/dogs-vs-cats/overview

# How to install

1.  After downloading the dataset, unzip it and form the following folder structure in your working directory: <code>data/dogs-vs-cats/</code>
2.  Inside the structure paste the images from the trainning.
3.  There are 25,000 images of dogs and cats (combined).
4.  Once the folder structure is created, you're good to go ahead and run the jupyter notebook.

# Data Preprocessing

The data is in format of images (to be specific: .jpg format). Hence it needs to be converted in form of an array of RGB values so that the model can understand. The preprocessing technique used here is similar to the VGG16 model.
It's basic intuitation is that, it takes the mean values of R,G,B (Red separate, blue separate and green separate) from the dataset and subtracts it from the dataset. After this the image is re-sized to 224 by 224.

## About the code

1. Python is used for building the model, trainning and prediction.
2. For trainning, I have enabled GPU computing for faster trainning process.

# Accuracy

The model gives a trainning accuracy of 83.68% and validation accuracu of 83.33%. The accuracy of test batch is around 81%. The metrics used for this classifier is **accuracy score**. Adam is used as an optimizer with a learning rate of 0.0001. **Adam** is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. **Adam** combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. The activation function used for each layer is RELU, except for the output layer which is softmax function.

During trainning, the number of epochs is set to 4 as the model starts overfitting if the model is trained for more number of epochs.
