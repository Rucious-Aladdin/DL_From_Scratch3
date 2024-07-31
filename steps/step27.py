if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph
import math

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    fact = 1
    for i in range(100000):
        if i != 0:
            fact *= (2*i * (2*i+1))
        #c = (-1) ** i / math.factorial(2 * i + 1)
        c = (-1) ** i / fact
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

if __name__ == "__main__":
    x = Variable(np.array(np.pi/4))
    y = sin(x)
    y.backward()

    print("Sin Function(np.cos()):")
    print(f"y.data: {y.data}")
    print(f"x.grad: {x.grad}")
    print()

    print("Sin Function(Taylor):")
    x = Variable(np.array(np.pi/4))
    y = my_sin(x, threshold=1e-150)
    y.backward()
    print(f"y.data: {y.data}")
    print(f"x.grad: {x.grad}")
    print()

    x.name = "x"; y.name="y"
    plot_dot_graph(y, verbose=False, to_file="step27_sin(Taylor_Series).png")