# PyTorch-DeepLearning
Breast Cancer Detection using PyTorch
This project implements a binary classification model to detect breast cancer (Malignant vs. Benign) using the Breast Cancer Wisconsin (Diagnostic) Dataset. It features two distinct PyTorch implementations to demonstrate the underlying mechanics of neural networks versus the standard high-level API usage.

📂 Project Structure
The project consists of two main notebooks:

PyTorch_training_pipeline_manual.ipynb

Concept: A "Low-Level" implementation.

Details: Manually defines weights/biases, calculates the forward pass using matrix multiplication, computes Binary Cross Entropy loss manually, and performs gradient descent updates without using an optimizer.

Goal: To understand the mathematics behind backpropagation and tensor operations.

PyTorch_training_pipeline_using_nn_Module.ipynb

Concept: A "High-Level" implementation (Industry Standard).

Details: Uses torch.nn.Module, nn.Linear, nn.BCELoss, and torch.optim.SGD.

Goal: To demonstrate scalable and modular deep learning model building.

📊 Dataset
The dataset is loaded directly from a raw GitHub URL within the notebooks.

Source: Breast Cancer Wisconsin (Diagnostic) Data Set.

Features: 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass (e.g., radius, texture, perimeter, area, smoothness).

Target: diagnosis (M = Malignant, B = Benign).

Preprocessing Steps:

Cleaning: Dropped unnecessary columns (id, Unnamed: 32).

Encoding: Converted target labels (M/B) to integers (1/0) using LabelEncoder.

Scaling: Standardized feature values using StandardScaler to ensure stable gradient descent.

🛠 Dependencies
To run this project, you will need the following Python libraries:

Python
import torch           # PyTorch for building the Neural Network
import pandas as pd    # Data manipulation
import numpy as np     # Numerical operations
import sklearn         # Data splitting and preprocessing
🧠 Model Architectures
1. Manual Implementation (MySimpleNN)
Weights: Initialized randomly using torch.rand (Double precision).

Bias: Initialized to zeros.

Forward Pass: sigmoid(X @ weights + bias)

Loss Function: Manual calculation of Log Loss (Binary Cross Entropy).

Python
loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()
2. PyTorch API Implementation (NeuralNet)
Parent Class: Inherits from nn.Module.

Layer: Single nn.Linear layer (Logistic Regression equivalent).

Activation: nn.Sigmoid.

Optimizer: Stochastic Gradient Descent (optim.SGD).

Loss Function: nn.BCELoss.

🚀 How to Run
Open either notebook in Google Colab or Jupyter Notebook.

Ensure your runtime supports Python 3.

Run the cells sequentially.

The data will automatically download and process.

The model will train for 50 epochs.

The final cell evaluates the model accuracy on the test set.

📈 Results
The models are trained for 50 epochs. Due to the small dataset size and simple architecture (Single Layer Perceptron / Logistic Regression), accuracies generally hover around 58% - 90% depending on hyperparameter tuning (learning rate) and random initialization.

Current default settings:

Learning Rate: 0.1 - 0.2

Epochs: 50

Threshold: 0.9 (for binary classification)

🤝 Contributing
Feel free to fork this project and improve the accuracy by:

Adding hidden layers (creating a Deep Neural Network).

Adjusting the learning rate or increasing epochs.

Implementing a different optimizer (e.g., Adam).
