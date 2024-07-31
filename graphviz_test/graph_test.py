if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph

if __name__ == "__main__":
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y = x0 + x1

    x0.name = "x0"
    x1.name = "x1"
    y.name = "y"

    txt = get_dot_graph(y, verbose=False)
    print(txt)

    with open("sample.dot", "w") as f:
        f.write(txt)