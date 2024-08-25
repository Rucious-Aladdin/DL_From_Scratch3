if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import os 
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

if __name__ == "__main__":
    max_epoch = 20
    batch_size = 100

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    model_path = "./models"
    model_name = "my_mlp.npz"
    path = os.path.join(model_path, model_name)

    if os.path.exists(path):
        model.load_weights(path)
    
    if dezero.cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()

    for epoch in range(max_epoch):
        sum_loss = 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)
        
        print("epoch: {}, loss: {:.4f}".format(epoch + 1, sum_loss / len(train_set)))
    
    model.save_weights(path)