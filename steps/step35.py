if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name, y.name = "x", "y"
    y.backward(create_graph= True)

    iters = 7

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
    
    gx = x.grad
    gx.name = "gx" + str(iters + 1)
    plot_dot_graph(gx, verbose=False, to_file=f"./viz_images/step35_tanh_{iters + 1}-th_derivative.png")