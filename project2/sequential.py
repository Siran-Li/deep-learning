import torch
import math
from module import Module

class Sequential(Module):
    def __init__(self, *layers):
        self.model = layers
        self.nb_layers = len(self.model)

    def forward(self, x):
        for layer in self.model:
            out = layer(x)
            x = out

        return out

    def backward(self, prev_grad):
        for i in range(self.nb_layers ):
            layer = self.model[self.nb_layers-1-i]
            grad = layer.backward(prev_grad)
            prev_grad = grad
        return grad
    
    
    def parameters(self):
        self.params = []
        for layer in self.model:
            for param in layer.parameters():
                self.params.append(param)

        return self.params