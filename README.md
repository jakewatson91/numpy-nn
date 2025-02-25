# Using Numpy to build a 2-Layer Softmax Neural Network to Classify MNIST Fashion Images
## Overview

This project implements a 2-layer softmax neural network to classify images from the Fashion MNIST dataset into 10 different categories (e.g., shoes, t-shirts, dresses, etc.).
The model is trained using stochastic gradient descent (SGD) to minimize a cross-entropy loss function with L2 regularization on weights.

## Dataset

The dataset consists of 28 × 28 pixel grayscale images, which are flattened into 784-dimensional vectors.
The labels correspond to one of 10 fashion categories.

## Download the Dataset

Use the following links to download Fashion MNIST:
	•	Train Images
	•	Train Labels
	•	Test Images
	•	Test Labels

## Loading the Data in Python

import numpy as np

train_images = np.load("fashion_mnist_train_images.npy")
train_labels = np.load("fashion_mnist_train_labels.npy")
test_images = np.load("fashion_mnist_test_images.npy")
test_labels = np.load("fashion_mnist_test_labels.npy")

print(f"Train Images Shape: {train_images.shape}")  # (n_train, 784)
print(f"Train Labels Shape: {train_labels.shape}")  # (n_train,)
print(f"Test Images Shape: {test_images.shape}")    # (n_test, 784)
print(f"Test Labels Shape: {test_labels.shape}")    # (n_test,)

## Model Architecture

The network consists of:
	•	Input Layer: 784-dimensional vector (flattened 28×28 image).
	•	Output Layer: 10 neurons with softmax activation (each representing a class probability).

Loss Function (Cross-Entropy with L2 Regularization)

f_CE(W, b) = - (1/n) Σ Σ y_k(i) log(ŷ_k(i)) + (α/2) Σ w_k^T w_k

### Where:
	•	n = number of examples
	•	α = regularization constant
	•	W = weight matrix
	•	b = bias vector
	•	ŷ_k = predicted probability for class k
	•	y_k = one-hot encoded true label

## Training Strategy
	•	Optimizer: Stochastic Gradient Descent (SGD)
	•	Regularization: L2 weight decay (applied only to weights, not biases).
	•	Hyperparameter Tuning:
	•	Learning rate
	•	Regularization strength (α)
	•	Batch size
	•	Dataset Split: The training data is split into training and validation sets.

## Training the Model with SGD

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(X, y, W, b, alpha):
    n = X.shape[0]
    logits = X @ W + b
    y_pred = softmax(logits)
    
    # Cross-entropy loss
    loss = -np.sum(y * np.log(y_pred + 1e-9)) / n
    
    # L2 Regularization
    reg_loss = (alpha / 2) * np.sum(W ** 2)
    
    return loss + reg_loss

### Hyperparameter Optimization

We optimize the same hyperparameters as in Homework 2 (Age Regression), testing at least 20 different combinations.

### Example grid search:

learning_rates = [0.01, 0.001, 0.0001]
regularization_strengths = [0.01, 0.1, 1.0]
batch_sizes = [32, 64, 128]

## Evaluation

After training, the model is evaluated on the test set using accuracy as the main metric.

def evaluate_model(X_test, y_test, W, b):
    logits = X_test @ W + b
    y_pred = np.argmax(softmax(logits), axis=1)
    accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {accuracy:.4f}")

## Results & Analysis
	•	Best hyperparameters found through tuning.
	•	Accuracy on test set after final training.

## References
	•	Fashion MNIST Dataset
	•	Goodfellow et al. (2016) – Deep Learning
	•	Bishop (2006) – Pattern Recognition and Machine Learning
