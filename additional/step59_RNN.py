if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.layers as L

if __name__ == "__main__":
    batch_size = 32
    rnn = L.RNN(10)
    x = np.random.rand(batch_size, 1)
    h = rnn(x)
    print(h.shape)
    