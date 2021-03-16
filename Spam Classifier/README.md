# Spam Classifier

<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="Pandas" src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

# Aim
To build a deep learning model to classifiy spam emails.

## Feature
It can classifiy SMS as spam and ham.

# Main Concept
### Neural Network
A **neural network** is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.
### Artificial NN
An **artificial neural network** (ANN) is the component of **artificial** intelligence that is meant to simulate the functioning of a human brain. Processing units make up ANNs, which in turn consist of inputs and outputs.

# Dataset
For this project, the dataset was downloaded from kaggle
the link: https://www.kaggle.com/uciml/sms-spam-collection-dataset
# About the dataset

 1. Contains around 6,000 rows.
 2. Train size is 4179 and test size is 1392 images
 3. Each image is of size 28 by 28.
 4. The dataset includes grayscale images.

# Data Preprocessing
**One hot encoding** was used to prepare the target variable.
CountVectorizer from sklearn was used to extract features from the data. It converts a collection of text documents to a matrix of token counts
This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.

## About the code
1. Python is used for building the model, trainning and prediction.
2. For trainning, I have enabled GPU computing for faster trainning process.

# Accuracy
The model gives a trainning accuracy of 100% and validation accuracy of 98.30%. The accuracy of test batch is 98.10%. The metrics used for this classifier is **accuracy score**. Adam is used as an optimizer with a learning rate of 0.0001. **Adam** is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. **Adam** combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. The activation function used for  layer were *relu* and *tanh* , except for the output layer which is *softmax* function. 





