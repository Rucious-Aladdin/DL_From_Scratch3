from step02 import Function, Square
from step01 import Variable
import numpy as np

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(.5)) # Variable 에 __call__을 계속 적용
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data) # y는 Variable 클래스
    print(type(y))