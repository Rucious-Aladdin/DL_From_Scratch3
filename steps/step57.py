if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import dezero
from dezero import Variable
import dezero.functions as F
import numpy as np

if __name__ == "__main__":
    N, C, H, W = 1, 5, 15, 15
    OC, (KH, KW) = 8, (3, 3)

    x = Variable(np.random.randn(N, C, H, W))
    W = np.random.randn(OC, C, KH, KW)
    y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
    y.backward()

    print(y.shape)
    print(x.grad.shape)
    """
    (1, 8, 15, 15)
    (1, 5, 15, 15)
    """