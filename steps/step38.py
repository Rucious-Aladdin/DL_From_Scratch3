if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F

if __name__ == "__main__":
    print("Using np.reshape:")
    x = np.array([[1, 2, 3], [4, 5, 6]]) # <- (2, 3)
    y = np.reshape(x, (6,))
    print(y) # <- (6, )
    print()

    print("USING RESHAPE(DeZero):")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,)) # <- F.reshape(x:Variable, shape:tuple)
    y.backward(retain_grad=True)
    print(x.grad) # 왜 grad가 1? -> 그대로 유지해서 넘기면 되기 때문!
    print()

    print("Using np.random.rand!")
    x = np.random.rand(1, 2, 3)
    y = x.reshape((2, 3)); print(y)
    y = x.reshape([2, 3]); print(y)
    y = x.reshape(2, 3); print(y)
    print()

    print("가변 인수 사용성 개선:")
    x = Variable(np.random.randn(1, 2, 3))
    y = x.reshape((2, 3)); print(y)
    y = x.reshape(2, 3); print(y)
    print()

    print("USING NumPy Transpose:")
    x = np.array([[1, 2, 3], [4, 5, 6]]); print(f"x:\n {x}") # <- (2, 3)
    y = np.transpose(x); print(f"y:\n {y}") # <- (3, 2)

    print("Dezero Transpose:")
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward()
    print(f"x.grad:\n{x.grad}")
    print()

    print("Variable Class내 추가 개선:")
    x = Variable(np.array(np.random.rand(2, 3)))
    y = x.transpose(); print(y)
    y = x.T; print(y)