if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Layer, Parameter
import numpy as np

if __name__ == "__main__":
    layer = Layer()

    l1 = Layer()
    l1.p1 = Parameter(np.array(1))

    layer.l1 = l1
    layer.p2 = Parameter(np.array(2))
    layer.p3 = Parameter(np.array(3))

    params_dict = {}
    layer._flatten_params(params_dict)
    print(params_dict)