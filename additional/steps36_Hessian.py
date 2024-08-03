if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
from dezero import Variable
import dezero.functions as F

def f(x):
    t = x ** 2
    y = F.sum(t)
    return y

if __name__ == "__main__":
    x, v = Variable(np.array([1.0, 2.0]), np.array([4.0, 5.0]))
    y = f(x)
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    ## 아직 작동안함. 나중에 해보기!
    z = F.matmul(v, gx)
    z.backward()
    print(x.grad)