from ai.Linear import Linear
from ai.utils import relu
import numpy as np

# No GPU offloading like pytorch *yet*

class DQN:
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.layers = [Linear(input_dim, hidden_dims[0])] # first layer is just input->first hidden
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(Linear(hidden_dims[i - 1], hidden_dims[i])) # add hidden layers
            
        self.layers.append(Linear(hidden_dims[-1], output_dim)) # add last hidden->output
        
    def forward(self, x): # feed an input through the network
        for i, layer in enumerate(self.layers):
            x = layer.forward(x) # feed the current data through the next layer
            if i < len(self.layers) - 1:
                x = relu(x) # apply ReLu activation except on final layer
        return x
    
    def load(self, other: 'DQN'): # copy weights and biases from another dqn (used for target net sync)
        assert len(self.layers) == len(other.layers), "Layer count mismatch"

        for self_layer, other_layer in zip(self.layers, other.layers):
            self_layer.weights = np.copy(other_layer.weights)
            self_layer.biases = np.copy(other_layer.biases)

    def clip_gradients(self, max_val): # clip gradients to prevent exploding updates during backprop
        for layer in self.layers:
            layer.clip_gradients(max_val)
            
    def backward(self, grad_output): # backprop
        for i in reversed(range(len(self.layers))):
            grad_output = self.layers[i].backward(grad_output)

    def update(self, lr): # apply gradient descent step using learning rate
        for layer in self.layers:
            layer.update(lr)