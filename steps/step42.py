if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import *
from dezero.utils import *
import dezero.functions as F
import matplotlib.pyplot as plt

def predict(x):
    y = F.matmul(x, W) + b
    return y

if __name__ == "__main__":
    # 실험용 dataset load
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)

    print("TOY_DATASET:")
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
    print()

    ## plot toy_dataset
    plt.figure(figsize=(10, 7))
    plt.title("toy dataset")
    plt.scatter(x, y)
    plt.savefig("./viz_images/step42_toydataset.png")
    #plt.show()
    
    # LinearRegression - with Gradient Descent
    print("LinearRegression... Ready") 
    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))
    print(f"W: {W.shape}")
    print(f"b: {b.shape}")
    print()

    print("SET HyperParameter for LR:")
    lr=.1
    iters = 1000
    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward() # <- loss: Variable(backward가 가능!)

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        #print(f"iters-{i + 1}, W.data = {W.data}, b.data = {b.data}")
    print()
    
    plt.figure(figsize=(10, 7))
    plt.title("toy dataset")
    plt.scatter(x, y)
    
    x_orig = np.linspace(0, 1.0, 100).reshape(-1, 1)
    y_orig = x_orig * W.data + b.data
    plt.plot(x_orig, y_orig, color="r")
    plt.savefig("./viz_images/step42_toydataset_AfterLR.png")
    plt.show()

    print("FINAL W and b:")
    print(f"W: {W}")
    print(f"b: {b}")

    """
    <TERMINAL OUTPUT>

    TOY_DATASET:
    x: (100, 1)
    y: (100, 1)

    LinearRegression... Ready
    W: (1, 1)
    b: (1,)

    SET HyperParameter for LR:

    FINAL W and b:
    W: variable([[1.93655202]])
    b: variable([5.55807954])
    """
