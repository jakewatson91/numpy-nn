# Using Numpy to build a 2-Layer Softmax Neural Network to Classify MNIST Fashion Images
## Overview

This project implements a 2-layer softmax neural network to classify images from the Fashion MNIST dataset into 10 different categories (e.g., shoes, t-shirts, dresses, etc.).
The model is trained using stochastic gradient descent (SGD) to minimize a cross-entropy loss function with L2 regularization on weights.

## Dataset

The dataset consists of 28 × 28 pixel grayscale images, which are flattened into 784-dimensional vectors.
The labels correspond to one of 10 fashion categories.

## Download the Dataset

Use the links provided in the notebook to download datasetes

## Loading the Data in Python

```import numpy as np

train_images = np.load("fashion_mnist_train_images.npy")
train_labels = np.load("fashion_mnist_train_labels.npy")
test_images = np.load("fashion_mnist_test_images.npy")
test_labels = np.load("fashion_mnist_test_labels.npy")

print(f"Train Images Shape: {train_images.shape}")  # (n_train, 784)
print(f"Train Labels Shape: {train_labels.shape}")  # (n_train,)
print(f"Test Images Shape: {test_images.shape}")    # (n_test, 784)
print(f"Test Labels Shape: {test_labels.shape}")    # (n_test,)
```

## Model Architecture

The network consists of:
	•	Input Layer: 784-dimensional vector (flattened 28×28 image).
	•	Output Layer: 10 neurons with softmax activation (each representing a class probability).
	•	Loss Function (Cross-Entropy with L2 Regularization)

## Where:  
- n = number of examples  
- alpha = regularization constant  
- w1, w2 = weight matrices 
- b1, b2 = bias vectors 
- y_pred = predicted probability for class  
- y = one-hot encoded true label  

## Training Strategy
- Optimizer: Stochastic Gradient Descent (SGD)
- Regularization: L2 weight decay (applied only to weights, not biases).
- Hyperparameter Tuning

### Hyperparameter Optimization

We optimize testing at 12 different combinations.

### Example grid search:

mini_batches = [32, 64, 128]
learning_rates = [1e-3, 1e-4]
epochs = [5, 10, 20]
l2_alphas = [1e-3, 1.0]
hidden_size = [20, 40]

## Evaluation

After training, the model is evaluated on the test set using validation loss and accuracy as the metrics.

## References
	•	Fashion MNIST Dataset
	•	Goodfellow et al. (2016) – Deep Learning
	•	Bishop (2006) – Pattern Recognition and Machine Learning
