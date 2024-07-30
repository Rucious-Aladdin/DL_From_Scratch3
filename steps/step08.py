import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.creator = None
        self.grad = None
    

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):

        # 처리해야 할 함수를 초기화
        funcs = [self.creator]
        while funcs: # funcs가 비어 있지 않으면 실행.
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
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

    x = Variable(np.array(.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
