# Traffic Sign Recognition

<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="Pandas" src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

# Aim

To build a deep learning model to classifiy traffic signs.

## Feature

It can classifiy 43 different types of traffic signs.

# Main Concept

### Convolutional Neural Network

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

# Dataset

For this project, **GTSRB** dataset was used.
dataset link --> https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# About the dataset

The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.

- Single-image, multi-class classification problem
- More than 40 classes
- More than 50,000 images in total
- Large, lifelike database

# Data Preprocessing

_OpenCV_ was used for preprocessing of images.

- **Preprocessing of Images** -
  Histogram Equalization- It is a method that improves the contrast in an image, in order to stretch out the intensity range.
  Working- - Equalization implies _mapping_ one distribution (the given histogram) to another distribution (a wider and more uniform distribution of intensity values) so the intensity values are spread over the whole range. - To accomplish the equalization effect, the remapping should be the cumulative distribution function.
  So to equalize, we first extract R,G,B values (spliting channels) from the images and pass these values individually in the histequalize() (a function in opencv-python [cv2].). Now we obtain new values of R,G,B for the given images. Then we merge these values to form a new image. - Working of merging in _OpenCV_-
  `cv2.merge` takes single channel images and combines them to make a multi-channel image
- **One hot encoding** was used to prepare the target variable.

## About the code

1. Python is used for building the model, trainning and prediction.
2. For trainning, I have enabled GPU computing for faster trainning process.

# Accuracy

The model gives a trainning accuracy of 100% when data was augmented. The accuracy of test batch is 93.3%. The metrics used for this classifier is **accuracy score**. Adam is used as an optimizer with a learning rate of 0.0001. **Adam** is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. **Adam** combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. The activation function used for each layer is RELU, except for the output layer which is softmax function.

# Architecture of the model

|          Name          | Filters | Activation | Kernel Size / pool size (for max pooling layers) | Padding | Dropout Probability (Only for dropout layer) |
| :--------------------: | :-----: | :--------: | :----------------------------------------------: | :-----: | :------------------------------------------: |
| conv2d_1 (Input layer) |   32    |    ReLu    |          (5,5) {input_shape=(32,32,3)}           | 'same'  |                                              |
|    max_pooling2d_1     |    -    |     -      |                      (2,2)                       |    -    |                                              |
|        conv2d_2        |   16    |    ReLu    |                      (5,5)                       | 'same'  |                                              |
| batch_normalization_1  |    -    |     -      |                        -                         |    -    |                                              |
|        conv2d_3        |   16    |    ReLu    |                      (5,5)                       | 'same'  |                                              |
| batch_normalization_2  |    -    |     -      |                        -                         |    -    |                                              |
|    max_pooling2d_2     |    -    |     -      |                      (2,2)                       |    -    |                                              |
|        conv2d_4        |   32    |    ReLu    |                      (5,5)                       | 'same'  |                                              |
| batch_normalization_3  |    -    |     -      |                        -                         |    -    |                                              |
|        conv2d_5        |   32    |    ReLu    |                      (5,5)                       | 'same'  |                                              |
| batch_normalization_4  |    -    |     -      |                        -                         |    -    |                                              |
|    max_pooling2d_3     |    -    |     -      |                      (2,2)                       |    -    |                                              |
|        flatten         |    -    |     -      |                        -                         |    -    |                                              |
|       dropout_1        |    -    |     -      |                        -                         |    -    |                     0.5                      |
|        dense_1         |   512   |    ReLu    |                        -                         |    -    |                                              |
|        dense_2         |   256   |    ReLu    |                        -                         |    -    |                                              |
| dense_3 (output layer) |   43    |  softmax   |                        -                         |    -    |                                              |
