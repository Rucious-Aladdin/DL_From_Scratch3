if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero.models import MLP
from dezero import Variable, as_variable
import dezero.functions as F

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y

if __name__ == "__main__":
    np.random.seed(0)

    print("MODEL: MLP")
    model = MLP((10, 3))
    x = np.array([[.2, -.4]])
    y = model(x)
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(y)
    print()

    print("Softmax1D:")
    x = Variable(np.array([[.2, -.4]]))
    y = model(x)
    p = softmax1d(y)
    print(y)
    print(p)
    print()

    print("Softmax1D:")
    x = Variable(np.array([[.2, -.4],
                           [1, 2],
                           [2, 3],
                           [3, 4]]))
    y = model(x)
    p = softmax_simple(y)
    print(y)
    print(p)
    print()

    print("Softmax_With_Cross_Entropy:")
    x = np.array([[0.2, -0.4],[0.3, 0.5],[1.3, -3.2],[2.1, 0.3]])
    t = np.array([2, 0, 1, 0])
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    print(loss)