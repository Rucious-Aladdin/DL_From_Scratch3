import numpy as np

if __name__ == "__main__":
    print("1개의 ndarray 저장")
    x = np.array([1, 2, 3])
    np.save("test.npy", x)

    x = np.load("test.npy")
    print(x)
    print()

    print("여러개의 ndarray 저장")
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])

    np.savez("test.npz", x1=x1, x2=x2)
    arrays = np.load("test.npz")
    print(type(arrays))
    print(arrays.keys())
    print(arrays["x1"])
    print(arrays["x2"])
    print()

    print("가변 인수로 여러개의 ndarray 저장")
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    data = {"x1": x1, "x2": x2}
    np.savez("test.npz", **data)

    arrays = np.load("test.npz")
    x1 = arrays["x1"]
    x2 = arrays["x2"]
    print(x1)
    print(x2)
    print()