if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
from step24 import goldstein

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    z.backward()

    x.name, y.name, z.name = "x", "y", "z"
    plot_dot_graph(z, verbose=False, to_file="goldstein.png")
