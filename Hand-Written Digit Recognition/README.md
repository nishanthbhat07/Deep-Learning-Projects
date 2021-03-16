# Hand-Written Digits Recognition

<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="Pandas" src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

# Aim
To build a deep learning model to classifiy hand written digits.

## Feature
It can classifiy images of digits from 0 to 9.

# Main Concept
### Convolutional Neural Network
A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

# Dataset
For this project, **MNIST** dataset was used.

# About the dataset

 1. Contains 70,000 images of 0 to 9 digits.
 2. Train size is 60,000 images and test size is 10,000 images
 3. Each image is of size 28 by 28.
 4. The dataset includes grayscale images.

# Data Preprocessing
**One hot encoding** was used to prepare the target variable.

## About the code
1. Python is used for building the model, trainning and prediction.
2. For trainning, I have enabled GPU computing for faster trainning process.

# Accuracy
The model gives a trainning accuracy of 98.76% and validation accuracy of 98.48%. The accuracy of test batch is 98.3%. The metrics used for this classifier is **accuracy score**. Adam is used as an optimizer with a learning rate of 0.0001. **Adam** is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. **Adam** combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. The activation function used for each layer is RELU, except for the output layer which is softmax function. 





