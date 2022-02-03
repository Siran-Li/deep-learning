import torch
from module import Module

class MSELoss(Module):
    """
    Implementation of: loss = (|| input - target ||^2).mean()
    
    Usage:
        pos1: input
        pos2: target
    """
    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, input, target):
        assert input.size() == target.size()

        if len(input.size()) == 1:
            input = input.reshape((1, input.size(0)))
            target = target.reshape((1, target.size(0)))

        self.input = input
        self.target = target

        loss = ((self.input - self.target)**2).mean()
        return loss

    def backward(self):
        self.grad = (self.input - self.target) * 2
        return self.grad / (self.target.size(0) * self.target.size(1))

    def parameters(self):
        pass


