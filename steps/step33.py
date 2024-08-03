if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
import copy

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


if __name__ == "__main__":
    # 순전파시 x.data, y.data 생성
    print("1st-Derivative:")
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(f"y.data: {y.data}")
    print(f"x.grad(1st-derivative): {x.grad}")
    print()

    print("2nd-Derivative:")
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(f"x.grad(2nd-derivative): {x.grad}")
    print()

    print("3nd-Derivative:")
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(f"x.grad(3nd-derivative): {x.grad}")
    print()

    ## Newton Method를 이용한 최적화
    x = Variable(np.array(2.0))
    iters = 10
    print(f"NEWTON OPTIMIZATION: iterations-{iters}")
    for i in range(iters):
        print(f"iters - {i}, x: {x}")
        
        y = f(x)
        x.cleargrad()
        # NOTE: 2차 미분을 구하기 위해 create_graph=True
        y.backward(create_graph=True) 

        gx = x.grad
        x.cleargrad()
        gx.backward()
        gx2 = x.grad

        # Newton Optimization Formula
        # NOTE: x <- x - (f'(x) / f''(x))
        x.data -= gx.data / gx2.data
