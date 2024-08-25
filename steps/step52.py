if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import dezero
import dezero.cuda
import dezero.datasets
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

if __name__ == "__main__":
    max_epoch = 5
    batch_size = 100

    train_set = dezero.datasets.MNIST(train=True)  
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    #dezero.cuda.gpu_enable = False
    if dezero.cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()
    
    for epoch in range(max_epoch):
        start = time.time()
        sum_loss = 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)
        
        elapased_time = time.time() - start
        print("epoch: {}, loss: {:.4f}, time: {:.4f}[sec]".format(epoch + 1, sum_loss / len(train_set), elapased_time))

        """
        <USING GPU>:
        epoch: 1, loss: 1.9173, time: 8.3602[sec]
        epoch: 2, loss: 1.2873, time: 4.1249[sec]
        epoch: 3, loss: 0.9253, time: 3.8974[sec]
        epoch: 4, loss: 0.7399, time: 3.9891[sec]
        epoch: 5, loss: 0.6350, time: 4.0696[sec]

        <USING CPU>:
        epoch: 1, loss: 1.9063, time: 39.3360[sec]
        epoch: 2, loss: 1.2762, time: 22.9417[sec]
        epoch: 3, loss: 0.9170, time: 22.1678[sec]
        epoch: 4, loss: 0.7339, time: 19.1184[sec]
        epoch: 5, loss: 0.6302, time: 29.8367[sec]

        """