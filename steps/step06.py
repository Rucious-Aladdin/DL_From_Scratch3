import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.input = None
    
class Function:
    def __call__(self, input):
        x =  input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        return gy * (2 * x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return gy * np.exp(x)

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    
    ## forward
    a = A(x)
    b = B(a)
    y = C(b)

    ## backward
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
    print(type(x))