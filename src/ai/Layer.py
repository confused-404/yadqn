class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def update_params(self, lr):
        pass # for layers with weights
