if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dezero
import numpy as np
import matplotlib.pyplot as plt
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# This is Preprocess Function
def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x

if __name__ == "__main__":
    print("Downloading the MNIST dataset...")
    train_set = dezero.datasets.MNIST(train=True, transform=f)
    test_set = dezero.datasets.MNIST(train=False, transform=f)
    print("Download is done!")
    print(len(train_set))
    print(len(test_set))
    print()

    print("Checking the first data...")
    x, t = train_set[0]
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.savefig("./viz_images/step51_mnist_sample.png")
    plt.show()
    print("Label:", t)
    print()

    ## Learning MNIST
    print("Start Learning MNIST...")
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = dezero.datasets.MNIST(train=True)
    test_set = dezero.datasets.MNIST(train=False)
    train_loader = dezero.DataLoader(train_set, batch_size) 
    test_loader = dezero.DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 10))
    optimizer = optimizers.SGD().setup(model)

    history = {"train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}
    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        
        print("epoch: {}".format(epoch+1))
        print("train loss: {:.4f}, accuracy: {:.4f}".format(sum_loss / len(train_set), sum_acc / len(train_set)))
        history["train_loss"].append(sum_loss / len(train_set))
        history["train_accuracy"].append(sum_acc / len(train_set))

        sum_loss, sum_acc = 0, 0
        with dezero.no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
        print("test loss: {:.4f}, accuracy: {:.4f}".format(sum_loss / len(test_set), sum_acc / len(test_set)))
        history["test_loss"].append(sum_loss / len(test_set))
        history["test_accuracy"].append(sum_acc / len(test_set))
    
    ## plot
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("spiral-train/test loss")
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["test_loss"], label="test loss")
    plt.legend(loc="best")
    
    plt.subplot(1, 2, 2)
    plt.title("spiral-train/test accuracy")
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["test_accuracy"], label="test accuracy")
    plt.legend(loc="best")
    
    plt.savefig("./viz_images/step51_MNIST_train_test_loss_acc.png")
    plt.show()