if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = x ** 2
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()

    ## (dy / dx) ** 3 + y 형태의 미분 문제도 풀수 있다!
    z = gx ** 3 + y
    z.backward()
    print(x.grad)