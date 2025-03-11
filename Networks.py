import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt-based backend (if installed)
import matplotlib.pyplot as plt
import time

# Basic Linear Regression using NumPy for a for-loop approach
class LinearRegressionBasicNumpyNet_For_Loop:
    def __init__(self, input_size=2, learning_rate=0.02, init_range=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(low=-init_range, high=init_range, size=(input_size, 1))
        self.biases = np.random.uniform(low=-init_range, high=init_range, size=1)

    def train(self, inputs, targets, epochs=100):
        observations = inputs.shape[0]
        for i in range(epochs):
            outputs = np.dot(inputs, self.weights) + self.biases
            deltas = outputs - targets
            loss = np.sum(deltas ** 2) / (2 * observations)
            print(f"Epoch {i + 1}, Loss: {loss}")
            deltas_scaled = deltas / observations
            weight_gradient = np.dot(inputs.T, deltas_scaled)
            bias_gradient = np.sum(deltas_scaled)
            self.weights -= self.learning_rate * weight_gradient
            self.biases -= self.learning_rate * bias_gradient

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def get_parameters(self):
        return self.weights, self.biases

# Linear Regression using Sklearn
class LR_Sklearn:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, inputs, targets):
        self.model.fit(inputs, targets)
        print("Training completed.")

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_parameters(self):
        return self.model.coef_, self.model.intercept_

# Simple Linear Regression with TensorFlow
class SLR_BasicTF:
    def __init__(self, input_size=2, output_size=1):
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')

    def train(self, training_inputs, training_targets, epochs=100, verbose=0):
        self.model.fit(training_inputs, training_targets, epochs=epochs, verbose=verbose)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_weights(self):
        return self.model.layers[0].get_weights()

# Advanced Linear Regression with TensorFlow
class SLR_AdvanceTF:
    def __init__(self, input_size=2, output_size=1):
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(output_size,
                                  kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                  bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), loss='mean_squared_error')

    def train(self, training_inputs, training_targets, epochs=100, verbose=0):
        self.model.fit(training_inputs, training_targets, epochs=epochs, verbose=verbose)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_weights(self):
        return self.model.layers[0].get_weights()

# TensorFlow Linear Regression in Tensor environment
class SLR_in_Tensor_env:
    def __init__(self, input_size=2, output_size=1, learning_rate=0.05, init_range=0.1):
        """
        Initialize the linear regression model using TensorFlow 2.x.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Define model variables (weights and biases) using tf.Variable
        self.weights = tf.Variable(tf.random.uniform([input_size, output_size], minval=-init_range, maxval=init_range), name="weights")
        self.biases = tf.Variable(tf.random.uniform([output_size], minval=-init_range, maxval=init_range), name="biases")

        # Define optimizer
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)

    def train(self, training_inputs, training_targets, epochs=100):
        """
        Train the model using the provided dataset.
        """
        # Cast the inputs and targets to float32 to match the data type of weights and biases
        training_inputs = tf.cast(training_inputs, tf.float32)
        training_targets = tf.cast(training_targets, tf.float32)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                outputs = tf.matmul(training_inputs, self.weights) + self.biases
                #loss_fn = tf.keras.losses.MeanSquaredError() # Mean Squared Error (MSE)
                loss_fn = tf.keras.losses.Huber() # Huber Loss
                loss = loss_fn(training_targets, outputs) / 2.0

            # Compute gradients and update parameters
            gradients = tape.gradient(loss, [self.weights, self.biases])
            self.optimizer.apply_gradients(zip(gradients, [self.weights, self.biases]))

            print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.6f}")

    def predict(self, test_inputs):
        """
        Generate predictions using the trained model.
        """
        test_inputs = tf.cast(test_inputs, tf.float32)  # Ensure test inputs are also cast to float32
        return tf.matmul(test_inputs, self.weights) + self.biases

    def get_parameters(self):
        """
        Retrieve the trained model parameters (weights and biases).
        """
        return self.weights.numpy(), self.biases.numpy()

# MNIST Classifier using TensorFlow
class MNISTClassifierTF2:
    def __init__(self, input_size=784, hidden_layer_size=50, output_size=10, learning_rate=0.001):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(output_size)  # logits output
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self.history = None

    def train(self, train_data, train_labels, validation_data, validation_labels, batch_size=100, max_epochs=15):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
        start_time = time.time()
        self.history = self.model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=(validation_data, validation_labels),
            callbacks=[early_stop],
            verbose=2
        )
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        return self.history

    def evaluate(self, test_data, test_labels):
        results = self.model.evaluate(test_data, test_labels, verbose=0)
        print(f"Test loss: {results[0]:.3f}, Test accuracy: {results[1] * 100:.2f}%")
        return results

    def plot_loss(self):
        if self.history:
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("Error: Train the model first!")

    def predict(self, inputs):
        return self.model.predict(inputs)