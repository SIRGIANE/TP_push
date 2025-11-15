import numpy as np
from .model import Model

class NeuralNetwork(Model):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a simple neural network with one hidden layer
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in the hidden layer
            output_size (int): Number of output classes
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X (np.array): Input data of shape (n_samples, input_size)
            
        Returns:
            tuple: (hidden layer output, output layer output)
        """
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a1, self.a2

    def backward(self, X, y, learning_rate=0.01):
        """
        Backward propagation
        
        Args:
            X (np.array): Input data
            y (np.array): True labels
            learning_rate (float): Learning rate for gradient descent
        """
        m = X.shape[0]
        
        # Output layer error
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer error
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Train the neural network
        
        Args:
            X (np.array): Training data
            y (np.array): Training labels
            epochs (int): Number of training iterations
            learning_rate (float): Learning rate for gradient descent
        """
        for _ in range(epochs):
            # Forward propagation
            self.forward(X)
            
            # Backward propagation
            self.backward(X, y, learning_rate)

    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predictions (0 or 1)
        """
        _, predictions = self.forward(X)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict probability estimates
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Probability estimates
        """
        _, predictions = self.forward(X)
        return predictions