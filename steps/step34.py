if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data]

    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(f"{i + 1}차 미분:")
        print(f"x.grad: {x.grad}")
        print()

    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data]

    for i in range(3):
        logs.append(x.grad.data)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(f"{i + 1}차 미분:")
        print(f"x.grad: {x.grad}")
        print()
    
    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc="lower right")
    plt.title("N-th_Derivatives(sin function)")
    plt.savefig("./viz_images/step34_N-th_Derivatives(sin function).png")
    plt.show()
