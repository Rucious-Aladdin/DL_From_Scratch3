if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import *
from dezero.utils import *
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Variable Instance VS Parameter Instance:")
    x = Variable(np.array(1.0))
    p = Parameter(np.array(2.0))
    y = x * p

    print(f"is p Parameter? {isinstance(p, Parameter)}")
    print(f"is x Parameter? {isinstance(x, Parameter)}")
    print(f"is y Parameter? {isinstance(y, Parameter)}")
    print()

    print("LINEAR CLASS:")
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print()

    x_fig, y_fig = x.reshape(-1).tolist(), y.reshape(-1).tolist()
    xy_cat = [(x_fig[i], y_fig[i]) for i in range(100)]
    xy_cat.sort(key=lambda x : x[0])
    x_fig = [xy_cat[i][0] for i in range(len(xy_cat))]
    y_fig = [xy_cat[i][1] for i in range(len(xy_cat))]
    plt.figure(figsize=(10, 7))
    plt.plot(x_fig, y_fig)
    
    l1 = L.Linear(5)
    l2 = L.Linear(1)

    def predict(x):
        y = l1(x)
        y = F.sigmoid(y)
        y = l2(y)
        return y

    lr = .2
    iters = 100000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data
        
        if i % 1000 == 0:
            print(f"iterations: {i} - loss: {loss.data}")
            lr *= 0.99
    
    y_pred = predict(x)
    y_pred_val = y_pred.data
    x_val = x
    x_fig, y_fig = x_val.reshape(-1).tolist(), y_pred_val.reshape(-1).tolist()
    xy_cat2 = [(x_fig[i], y_fig[i]) for i in range(100)]
    xy_cat2.sort(key=lambda x : x[0])
    x_fig2 = [xy_cat2[i][0] for i in range(len(xy_cat))]
    y_fig2 = [xy_cat2[i][1] for i in range(len(xy_cat))]
    plt.plot(x_fig2, y_fig2)
    plt.savefig("./viz_images/step44_Linear_Layer.png")
    plt.show()

    print(l1.W.data.shape)
    print(l2.W.data.shape)

