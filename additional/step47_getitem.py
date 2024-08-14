if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F

if __name__ == "__main__":
    print("BASIC USING")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.get_item(x, 1)
    print(y)
    y.backward()
    print(x.grad)
    print()

    print("Using Multiple Index:")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    indices = np.array([0, 0, 1])
    y = F.get_item(x, indices)
    print(y)
    print()

    ## __getitem__
    print("__getitem__ method")
    Variable.__getitem__ = F.get_item
    y = x[1]
    print(y)

    y = x[:, 2]
    print(y)

