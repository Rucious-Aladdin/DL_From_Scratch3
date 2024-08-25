if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_set = dezero.datasets.SinCurve(train=True)
    print(len(train_set))
    print(train_set[0])
    print(train_set[1])
    print(train_set[2])
    """
    num_samples = 300
    xs = [example for example in train_set.data[:num_samples]]
    ts = [example for example in train_set.data[1:num_samples + 1]]
    """

    xs = [example[0] for example in train_set]
    ts = [example[1] for example in train_set]
    plt.plot(np.arange(len(xs)), xs, label="xs")
    plt.plot(np.arange(len(ts)), ts, label="ts")
    plt.legend()
    plt.show()
