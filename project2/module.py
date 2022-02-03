
class Module(object):
    """
    Base Class. Other modules should inherit from it,
        and rewrite the ``forward'', ``backword'' and ``param'' functions.
    """
    def __call__(self, *input):
        return self.forward(*input)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def parameters(self):
        return []