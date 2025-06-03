from ai.Layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros((1, output_dim))

    def forward(self, input):
        self.input = input
        return input @ self.weights + self.biases