if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F

if __name__ == "__main__":
    print(f"Sin Function: np.array([[1, 2, 3], [4, 5, 6]])")
    x = Variable(np.array([[1, 2, 3],
                           [4, 5, 6]]))
    y = F.sin(x)
    print(y)
    print()

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    print(f"Add Function: x: {x.shape}, y: {y.shape}")
    y = x + c
    print(y)

    print("SUM Function:")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    t = x + c
    """
    # y = F.sum(t) <- F.sum() will be implemented at step39.
    y.backward(retain_grad=True)
    print(y.grad)
    print(t.grad)
    print(x.grad)
    print(c.grad)
    print()
    """