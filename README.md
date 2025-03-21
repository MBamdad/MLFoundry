# MLFoundry

MLFoundry is a machine learning project designed to simplify model development, and deployment, offering tools and scripts for building, training, and evaluating models across various tasks, including Linear Regression, Basic Neural Networks, Deep Learning (TensorFlow), and the MNIST dataset.

## Goals:
- Provide reusable pipelines for common ML tasks.
- Experiment with various models and algorithms.
- Facilitate reproducibility in scientific machine learning.

# Description:
The objective of this project is to implement a Simple Linear Regression model using the numpy and sklearn libraries, as well as an advanced Linear Regression model using TensorFlow, enabling efficient model training and optimization.
The project demonstrates the fundamental steps in machine learning, including creating datasets, building a model, training it, and evaluating the performance.
It applies TensorFlow’s Sequential model, which stacks layers for direct computation of the output via a linear transformation of the inputs (dot product with weights plus bias).


# Key features include:
Data Creation: 
Random datasets for training, stored as inputs and targets.

Model Building: 
A simple linear regression model using a single dense layer that computes the dot product of inputs and weights and adds the bias.
Optimization and Training: Use of an optimization algorithm like stochastic gradient descent (SGD) to minimize the loss function (Mean Squared Error).

Hyperparameter Tuning:
 Adjustable parameters such as learning rate, weight and bias initialization, allowing for a closer match to a traditional NumPy-based linear regression model.
Use Cases:
This approach is ideal for tasks involving regression where the goal is to predict a continuous output based on one or more input features. It can be used in areas such as:

Predicting house prices based on features like size and location.
Estimating stock prices based on historical data.
General prediction problems where a linear relationship between inputs and outputs is assumed.

Scientific and Real-World Problems Solved:
This project showcases how TensorFlow can be used to solve simple regression problems, providing an accessible introduction to machine learning. 
It aids in understanding the transition from traditional statistical models to machine learning frameworks like TensorFlow, allowing for easier scalability and optimization when moving to more complex neural networks in the future.

#  Install Packages from requirements.txt:
 - pip install -r requirements.txt

## Installation

To get started with the MLFoundry project, follow these steps:
### Using Conda
1. Create a new Conda environment: 
    conda create --name MLFoundry python=3.8
2. Activate the environment: 
    conda activate MLFoundry
3. Install dependencies:
    conda install --file requirements.txt
* Make sure to install the appropriate versions of Python (e.g., Python 3.8 or 3.9).
