from ai.Layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros((1, output_dim))
        # self.input
        # self.grad_weights
        # self.grad_biases

    def forward(self, input):
        self.input = input
        return input @ self.weights + self.biases
    
    def clip_gradients(self, max_val):
        np.clip(self.grad_weights, -max_val, max_val, out=self.grad_weights)
        np.clip(self.grad_biases, -max_val, max_val, out=self.grad_biases)

    def backward(self, grad_output):
        # grad_output shape (batch_size, output_dim)
        
        self.grad_weights = self.input.T @ grad_output # (input_dim, output_dim)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True) # (1, output_dim)
        grad_input = grad_output @ self.weights.T # (batch_size, input_dim)
        
        return grad_input
    
    def update(self, lr):
        self.weights -= lr * self.grad_weights
        self.biases -= lr * self.grad_biases