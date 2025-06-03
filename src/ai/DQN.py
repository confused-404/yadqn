from ai.Linear import Linear
from ai.utils import relu
import numpy as np

# No GPU offloading like pytorch *yet*

class DQN:
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.layers = [Linear(input_dim, hidden_dims[0])]
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(Linear(hidden_dims[i - 1], hidden_dims[i]))
            
        self.layers.append(Linear(hidden_dims[-1], output_dim))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = relu(x)
        return x
    
    def load(self, other: 'DQN'):
        assert len(self.layers) == len(other.layers), "Layer count mismatch"

        for self_layer, other_layer in zip(self.layers, other.layers):
            self_layer.weights = np.copy(other_layer.weights)
            self_layer.biases = np.copy(other_layer.biases)
