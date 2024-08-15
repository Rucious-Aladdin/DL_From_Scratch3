if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dezero
import numpy as np
import matplotlib.pyplot as plt

import math
import numpy as np
from dezero import optimizers
from dezero import transforms
import dezero.functions as F
from dezero.models import MLP

if __name__ == "__main__":
    print("Spiral dataset")
    train_set = dezero.datasets.Spiral(train=True)
    print(train_set[0])
    print(len(train_set))
    print()

    print("Spiral dataset (batch)")
    train_set = dezero.datasets.Spiral()
    batch_index = [0, 1, 2]
    batch = [train_set[i] for i in batch_index]
    print(len(batch))
    print(batch)
    print()

    # Hyperparameters
    max_epoch = 2000
    batch_size = 15
    hidden_size = 10
    lr = 1.0

    # Dataset, Model, Optimizer
    f = transforms.Normalize(mean=0.0, std=2.0)
    #f = None
    train_set = dezero.datasets.Spiral(transform=f)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    # Learning
    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    train_losses = []
    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            batch_index = index[i * batch_size:(i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)
        
        avg_loss = sum_loss / data_size
        print("epoch %d, loss %.5f" % (epoch + 1, avg_loss))
        train_losses.append(avg_loss)
    
    x = train_set.data
    t = train_set.label
    plt.clf()
    plt.title("Result Visualization")
    plt.plot(x[t == 0][:, 0], x[t == 0][:, 1], "o", label="t=0")
    plt.plot(x[t == 1][:, 0], x[t == 1][:, 1], "x", label="t=1")
    plt.plot(x[t == 2][:, 0], x[t == 2][:, 1], "s", label="t=2")

    # plot model decision boundary
    h = 0.001
    x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    # 모델 예측
    y_pred = model(X)  # 소프트맥스 출력 형태의 예측 결과
    y_pred_class = np.argmax(y_pred.data, axis=1)  # 각 포인트에 대해 가장 높은 확률을 가진 클래스를 선택
    y_pred_class = y_pred_class.reshape(xx.shape)  # 그리드 형태로 재구성

    # 결정 경계 시각화
    plt.contourf(xx, yy, y_pred_class, alpha=0.3, cmap=plt.cm.coolwarm)

    plt.legend(loc="best")
    plt.savefig("./viz_images/step49_result_visualization.png")
    plt.show()

