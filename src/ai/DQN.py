from ai.Linear import Linear
from ai.utils import relu


class DQN:
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.layers = [Linear(input_dim, hidden_dims[0])]
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(Linear(hidden_dims[i - 1], hidden_dims[i]))
            
        self.layers.append(Linear(hidden_dims[-1], output_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = relu(layer.forward(x))
        return x