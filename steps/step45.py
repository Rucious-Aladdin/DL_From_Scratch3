if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Layer, Model, Variable
from dezero.models import MLP

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

if __name__ == "__main__":
    model = Layer()
    model.l1 = L.Linear(5)
    model.l2 = L.Linear(3)
    def predict(model, x):
        y = model.l1(x)
        y = F.sigmoid(y)
        y = model.l2(y)
        return y
    for p in model.params():
        print(p)
    model.cleargrads()
    print()

    x = Variable(np.random.randn(5, 10), name="x")
    model = TwoLayerNet(100, 10)
    model.plot(x)
    print()

    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = .2
    max_iter = 10000
    hidden_size = 10

    model = MLP((10, 20, 30, 40, 1))
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(f"iterations - {i} - loss: {loss}")
    model.plot(x)