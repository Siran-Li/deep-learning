import torch
import math
from module import Module

class ReLU(Module):
    def __init__(self):
        self.type = 'ReLU'

    def forward(self, x):
        self.x = x
        out = x.clone()
        out[self.x <=0] = 0
        return out

    def backward(self, dldy):
        self.dsdx = torch.empty(self.x.size()).fill_(0)
        self.dsdx[self.x>0] = 1
        self.dldx = self.dsdx * dldy
        return self.dldx

    def parameters(self):
        return []

class Tanh(Module):
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, x):
        self.x = x
        # y = (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
        out = x.apply_(lambda y: (2 / (1 + math.exp(-2 * y))) - 1)
        return out

    def backward(self, dldy):
        #return 4 * ((self.x.exp() + self.x.mul(-1).exp()).pow(-2)) * dldy
        dydx = 1 - self.forward(self.x) ** 2
        dldx = dydx * dldy
        return dldx

    def parameters(self):
        return []


class Sigmoid(Module):
    def __init__(self):
        self.type = 'Sigmoid'
    
    def forward(self, x):
        self.x = x
        out = 1.0 / (1.0 + (-self.x).exp())
        return out

    def backward(self, dldy):
        sig_x = self.forward(self.x)
        self.dydx = sig_x * (1-sig_x)

        self.dldx = dldy * self.dydx
        return self.dldx
    
    def parameters(self):
        return []


class SELU(Module):
    def __init__(self) -> None:
        self.type = 'SELU'
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        self.x = x
        p1 = x.clone()
        p1[self.x <= 0] = 0

        p2 = self.alpha * (self.x.exp() - 1)
        p2[self.x > 0] = 0

        out = self.scale * (p1 + p2)
        return out

    def backward(self, dldy):
        dsdx_p1 = torch.empty(self.x.size()).fill_(1)
        dsdx_p1[self.x <=0] = 0

        dsdx_p2 = self.alpha * self.x.exp()
        dsdx_p2[self.x > 0] = 0

        self.dsdx = self.scale * (dsdx_p1 + dsdx_p2)
        self.dldx = self.dsdx * dldy
        return self.dldx

    def parameters(self):
        return []


if __name__ == "__main__":
    m = Sigmoid()

    x = torch.tensor([[-1.2, 3.2, 4.2], 
                      [2.1, -2.3, -2.4]])
    t = torch.tensor([[1.0, 1.0, 1.0,], 
                      [2.3, 2.3, 2.3]])
    
    out = m(x)
    print("out: ")
    print(out)
    
    from loss import MSELoss
    criterion = MSELoss()

    loss = criterion(out, t)
    print("loss: ")
    print(loss)

    dldy = criterion.backward()
    print("dldy: ")
    print(dldy)
    dldx = m.backward(dldy)
    print("x grad:")
    print(dldx)