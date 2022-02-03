import torch
import math
from module import Module

class Linear(Module):
    """
    Implementation of: y = W^T x + b
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False'', the layer will not learn an additive bias.
            Default: True

    Initialization: uniform distribution from U(-sqrt(k), sqrt(k)),
        where k = 1 / in_features
    """
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        
        self.k = 1 / self.in_features
        self.weight = torch.empty((self.in_features, self.out_features)).uniform_(-math.sqrt(self.k), math.sqrt(self.k))

        self.bias = None
        if bias:
            self.bias = torch.empty(self.out_features).uniform_(-math.sqrt(self.k), math.sqrt(self.k))

        self.x = None

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.reshape((1, x.size(0)))
        
        self.x = x           # Nx2
        out = self.x.mm(self.weight)
        if self.bias is not None:
            out += self.bias
        return out

    def backward(self, dldy):
        self.weight.grad = self.x.T.mm(dldy)
        self.bias.grad = dldy.sum(dim=0)
        self.dldx = dldy.mm(self.weight.T)

        return self.dldx

    def parameters(self):
        return [self.weight, self.bias]