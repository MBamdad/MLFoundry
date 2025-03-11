import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from Networks import (
    LinearRegressionBasicNumpyNet_For_Loop,
    LR_Sklearn,
    SLR_BasicTF,
    SLR_AdvanceTF,
    SLR_in_Tensor_env,
    MNISTClassifierTF2
)


def train_model(model_name, inputs, targets, val_inputs=None, val_targets=None):
    """
    Train the selected model. Supports both regression and MNIST classification models.

    Args:
        model_name (str): The name of the selected model.
        inputs (np.array): Training inputs.
        targets (np.array): Training targets.
        val_inputs (np.array, optional): Validation inputs (only for MNISTClassifierTF2).
        val_targets (np.array, optional): Validation targets (only for MNISTClassifierTF2).

    Returns:
        Trained model instance.
    """
    selected_model = None
    start_time = time.time()

    if model_name == 'LR_For_Loop':
        model = LinearRegressionBasicNumpyNet_For_Loop(input_size=2, learning_rate=0.02)
        model.train(inputs, targets, epochs=100)
        selected_model = model
    elif model_name == 'LR_Sklearn':
        model = LR_Sklearn()
        model.train(inputs, targets)
        selected_model = model
    elif model_name == 'LR_TF_Basic':
        model = SLR_BasicTF(input_size=2, output_size=1)
        model.train(inputs, targets, epochs=50)
        selected_model = model
    elif model_name == 'LR_TF_Advance':
        model = SLR_AdvanceTF(input_size=2, output_size=1)
        model.train(inputs, targets, epochs=50)
        selected_model = model
    elif model_name == 'SLR_in_Tensor_env':
        model = SLR_in_Tensor_env()
        model.train(inputs, targets, epochs=1000)
        selected_model = model
    elif model_name == 'MNISTClassifierTF2':
        model = MNISTClassifierTF2()
        model.train(inputs, targets, val_inputs, val_targets)  # Pass validation data
        selected_model = model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    return selected_model


def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        Tuple of numpy arrays: (train_images, train_labels, val_images, val_labels, test_images, test_labels).
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and reshape the images
    train_images = train_images.reshape(-1, 784).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 784).astype('float32') / 255.0

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)

    # Split validation set from training data
    val_images, val_labels = train_images[-10000:], train_labels[-10000:]
    train_images, train_labels = train_images[:-10000], train_labels[:-10000]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def train_mnist_classifier():
    """
    Load MNIST data, train and evaluate the classifier.
    """
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_data()
    classifier = MNISTClassifierTF2()
    classifier.train(train_images, train_labels, val_images, val_labels)
    classifier.evaluate(test_images, test_labels)
    classifier.plot_loss()
