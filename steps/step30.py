if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
import copy

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


if __name__ == "__main__":
    # 순전파시 x.data, y.data 생성
    print("FORWARD:")
    x = Variable(np.array(1.0))
    y = sin(x)
    print(f"x.data: {x.data}")
    print(f"x.grad: {x.grad}")
    print(f"y.data: {y.data}")
    print(f"y.grad: {y.grad}")
    print()


    # 역전파시 x.grad, y.grad 생성
    print("BACKWARD:")
    x = Variable(np.array(1.0))
    y = sin(x)
    y.backward(retain_grad=True)
    print(f"x.data: {x.data}")
    print(f"x.grad: {x.grad}")
    print(f"y.data: {y.data}")
    print(f"y.grad: {y.grad}")
    print()