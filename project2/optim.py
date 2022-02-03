import torch

class SGD(object):
    def __init__(self, params, lr):
        self.params = params    # a list
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = torch.empty(param.size()).fill_(0.)

    def step(self):
        for param in self.params:
            #add1 = hex(id(param))
            
            #===========================================
            # can only use -= operator
            # if use the second line, param will lose its reference 
            # to the original param, and the update will not work

            param -= self.lr * param.grad
            
            #param = param - self.lr * param.grad
            #===========================================

            #add2 = hex(id(param))
            #print(add1 == add2)