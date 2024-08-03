if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

if __name__ == "__main__":
    print("RosenBrock:")
    x0 = Variable(np.array(0.0)); x0.name = "x0"
    x1 = Variable(np.array(2.0)); x1.name = "x1"

    y = rosenbrock(x0, x1); y.name = "y"
    y.backward()
    print(f"x0.grad: {x0.grad:.3f}")
    print(f"x1.grad: {x1.grad:.3f}")

    lr = .001
    iters = 50000
    x0s = []
    x1s = []
    PrintVariable = True
    for i in range(iters):
        if PrintVariable:
            print(f"iters: {i:<3} - {x0}, {x1}")

        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

        x0s.append(x0.data.item())
        x1s.append(x1.data.item())
    
    ## Plot RosenBrock Function - Gradient Descent
    xx = np.linspace(-4, 4, 800)
    yy = np.linspace(-3, 3, 600)
    X, Y = np.meshgrid(xx, yy)
    Z = rosenbrock(X, Y)
    levels=np.logspace(-1, 3, 10)
    plt.contourf(X, Y, Z, alpha=0.1, levels=levels)
    plt.contour(X, Y, Z, colors="gray",
                levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000])
    plt.plot(1, 1, '*', markersize=10)
    plt.plot(x0s, x1s, marker="o", markersize=7, alpha=.3, color="green")
    plt.xlim(-2, 2)
    plt.ylim(-1, 3)
    plt.savefig(f"./viz_images/step28_rosenbrock_iters_{iters}.png")
    plt.show()