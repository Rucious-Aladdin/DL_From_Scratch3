if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from dezero import Model
import dezero.functions as F
import dezero.layers as L
import numpy as np
import dezero

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

if __name__ == "__main__":

    print("SIMPLE RNN TEST")
    seq_data = [np.random.randn(1, 1) for _ in range(1000)]
    xs = seq_data[0:-1]
    ts = seq_data[1:]

    model = SimpleRNN(10, 1)

    loss, cnt = 0, 0
    for x, t in zip(xs, ts):
        y = model(x)
        loss += F.mean_squared_error(y, t)
        cnt += 1
        if cnt == 3:
            model.cleargrads()
            loss.backward()
            break
    
    model.plot(x, to_file="./viz_images/step59_rnn.png")
    print("done")
    print()

    print("RNN TEST")
    max_epoch = 100
    hidden_size = 100
    bptt_length = 30
    train_set = dezero.datasets.SinCurve(train=True)
    seq_len = len(train_set)
    optimizer = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in train_set:
            x = x.reshape(1, 1)
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            if count == seq_len or count % bptt_length == 0:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
        avg_loss = float(loss.data) / count
        print(f"| epoch {epoch+1} | loss {avg_loss}")
    
    print("Plotting...")
    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data))
    

    plt.plot(np.arange(len(xs)), xs, label="y=cos(x)")
    plt.plot(np.arange(len(xs)), pred_list, label="predict")
    plt.title("y=cos(x) vs predict using RNN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("./viz_images/step59_sin_predict.png")
    plt.show()