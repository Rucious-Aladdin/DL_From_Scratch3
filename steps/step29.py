if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
import copy

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 - 4

if __name__ == "__main__":
    iter_GD = 1000; lr =0.01
    iter_newton = 10

    Gds = []
    x1 = Variable(np.array(2.0))
    for i in range(iter_GD):
        Gds.append(copy.deepcopy(x1.data))
        y1 = f(x1)
        x1.cleargrad()
        y1.backward()
        x1.data -= lr * x1.grad
   
    newtons = []
    x2 = Variable(np.array(2.0))
    for i in range(iter_newton):
        newtons.append(copy.deepcopy(x2.data))
        y2 = f(x2)
        x2.cleargrad()
        y2.backward()
        
        x2.data -= x2.grad / gx2(x2.data)
    
    x_original = np.linspace(-2.2, 2.2, 100)
    y_original = f(x_original)

    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.title("GD")
    plt.plot(x_original, y_original)
    print(Gds)
    x1s = np.array(Gds)
    y1s = f(x1s)
    plt.plot(x1s, y1s, marker="o", markersize=3)
    plt.ylim(-2, 10)

    plt.subplot(1, 2, 2)
    plt.title("newton")
    plt.plot(x_original, y_original)
    x2s = np.array(newtons)
    y2s = f(x2s)
    plt.plot(x2s, y2s, marker="o", markersize=3)
    plt.ylim(-2, 10)
    plt.savefig("./viz_images/step29_GD_Versus_Newton.png")
    plt.show()