from module import Module
from linear import Linear
from activation import ReLU, Tanh, Sigmoid, SELU
from sequential import Sequential


class MLP(Module):

    def __init__(self):
        self.mlp = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 1),
            Sigmoid()
        )

    def forward(self, x):
        out = self.mlp(x)
        return out

    def backward(self, prev_grad):
        self.grad = self.mlp.backward(prev_grad)
        return self.grad

    def parameters(self):
        self.params = self.mlp.parameters()
        return self.params