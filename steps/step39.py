if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F

if __name__ == "__main__":
    print("Dezero Addfunc, Broadcast:")
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)
    print()

    print("Dezero 2D-Array, Addfunc:")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)
    print()

    print("Dezero Sum-Backward:")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    print("Dezero Sum-4D array")
    x = Variable(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    print(y.shape)