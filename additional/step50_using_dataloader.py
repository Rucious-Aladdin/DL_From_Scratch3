if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero import DataLoader
from dezero.datasets import Spiral
import numpy as np
import dezero.functions as F

class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()
        self.cnt += 1
        return self.cnt

if __name__ == "__main__":

    # what is iterator?
    print("iterator basic:")
    t = [1, 2, 3]
    x = iter(t)
    print(next(x))
    print(next(x))
    print(next(x))
    print()

    print("MyIterator:")
    obj = MyIterator(5)
    for x in obj:
        print(x)
    print()
    
    ## DataLoader
    batch_size = 10
    max_epoch = 1

    train_set = Spiral(train=True)
    test_set = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    for epoch in range(max_epoch):
        for x, t in train_loader:
            print("train - x.shape, t.shape", end=" ")
            print(x.shape, t.shape)
            break

        for x, t in test_loader:
            print("test - x.shape, t.shape", end=" ")
            print(x.shape, t.shape)
            break

    ## accuracy
    y = np.array([[.2, .8, 0], [.1, .9, 0], [.8, .1, .1]])
    t = np.array([1, 2, 0])
    acc = F.accuracy(y, t)
    print(acc)
