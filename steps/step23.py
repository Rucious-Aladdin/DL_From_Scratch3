if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = (x + 3) ** 2
    y.backward()

    print(f"y: {y}")
    print(f"x.grad: {x.grad}")