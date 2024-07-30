import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() # backward 메서드의 재귀 호출
    
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

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.0)
    
    """
    Function은 input, output을 참조하는 양방향 연결
    Variable은 자신을 만들어낸 함수를 참조하는 단방향 연결
    """

    ## dy/db의 계산
    C = y.creator ## y (Variable Type) -> creator (Function Type) 
    b = C.input ## C (Function Type) -> input (Variable Type)
    b.grad = C.backward(y.grad) 
    
    ## dy/da == (dy/db) * (db/da) 의 계산
    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    ## dy/dx == (dy/db) * (db/da) * (da/dx)의 계산
    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)

    print(x.grad)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)