if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Layer, Model, Variable
from dezero.models import MLP
from dezero import optimizers
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = .2
    max_iter = 10000
    hidden_size = 10

    model = MLP((hidden_size, 1))
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)
    print("SGD:")
    sgd_loss = []
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print(loss)
            sgd_loss.append(float(loss.data))
    # - END OF SGD -

    #=================================================
    # MomentumSGD
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = .2
    max_iter = 10000
    hidden_size = 10

    model = MLP((hidden_size, 1))
    optimizer = optimizers.MomentumSGD(lr=.05, momentum=.9)
    optimizer.setup(model)
    print("Momentum:")
    momem_loss = []
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print(loss)
            momem_loss.append(float(loss.data))
    
    xs = [x * 1000 for x in range(len(sgd_loss))]
    plt.figure(figsize=(10, 7))
    plt.plot(xs, sgd_loss, marker="s", label="SGD")
    plt.plot(xs, momem_loss, marker="o", label="MomentumSGD")
    plt.legend()
    plt.savefig("./viz_images/step46_SGD_VS_MomentumSGD.png")
    plt.show()