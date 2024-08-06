if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import *
from dezero.utils import *
import dezero.functions as F
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    I, H, O = 1, 10, 1
    W1 = Variable(0.01 * np.random.randn(I, H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.randn(H, O))
    b2 = Variable(np.zeros(O))

    def predict(x):
        y = F.linear(x, W1, b1)
        y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        return y
    
    lr = .2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        W2.data -= lr * W2.grad.data
        b1.data -= lr * b1.grad.data
        b2.data -= lr * b2.grad.data
        if i % 1000 == 0:
            print(f"iters-{i}: {loss}")

    plt.figure(figsize=(10, 7))
    plt.title("ShallowNetwork(Sin_Graph)")
    plt.scatter(x, y)
    x_orig = np.linspace(0, 1.0, 100).reshape(-1, 1)
    y_orig = predict(x_orig)
    plt.plot(x_orig, y_orig.data, color="r")
    plt.savefig("./viz_images/step43_ShallowNetowk.png")
    plt.show()
