if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
from dezero.utils import sum_to
import dezero.functions as F

if __name__ == "__main__":
    print("Sum_to -> sum_to(target_shape)")
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y= sum_to(x, (1, 3))
    print(y)
    y = sum_to(x, (2, 1))
    print(y)
    print()

    print("BroadCast")
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    y.backward()
    print(x0.grad)
    print(x1.grad)

    """
    #<Before Implementation>
    print("BroadCast")
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    y.backward()
    print(x0.grad)
    print(x1.grad) # not correct!
    #BroadCast
    #variable([1 1 1])
    #variable([1 1 1])
    """
