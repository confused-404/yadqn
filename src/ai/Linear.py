from ai.Layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim) # the weights of each node, shape (input_dim, output_dim)
        self.biases = np.zeros((1, output_dim)) # biases of each node shape: (1, output_dim)
        # self.input
        # self.grad_weights
        # self.grad_biases

    def forward(self, input):
        self.input = input
        return input @ self.weights + self.biases # multiply input matrix by weights and add biases
    
    def clip_gradients(self, max_val):
        np.clip(self.grad_weights, -max_val, max_val, out=self.grad_weights)
        np.clip(self.grad_biases, -max_val, max_val, out=self.grad_biases)

    def backward(self, grad_output):
        # grad_output: gradient of loss with respect to output of this layer, shape (batch_size, output_dim)
        
        self.grad_weights = self.input.T @ grad_output # shape (input_dim, output_dim)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True) # (1, output_dim)
        grad_input = grad_output @ self.weights.T # grad_output for the next layer (batch_size, input_dim)
        
        return grad_input
    
    def update(self, lr): # update weights and biases with grad_weights and grad_biases and learning rate
        self.weights -= lr * self.grad_weights
        self.biases -= lr * self.grad_biases