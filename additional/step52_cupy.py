import cupy as cp
import numpy as np

if __name__ == "__main__":
    x = cp.arange(6).reshape(2, 3)
    print(x)

    y = x.sum(axis=1)
    print(y)

    # Numpy To Cupy
    n = np.array([1, 2, 3])
    c = cp.asarray(n)
    print(c, type(c))
    assert type(c) == cp.ndarray

    x = np.array([1, 2, 3])
    xp = cp.get_array_module(x)
    assert xp == np

    x = cp.array([1, 2, 3])
    xp = cp.get_array_module(x)
    assert xp == cp