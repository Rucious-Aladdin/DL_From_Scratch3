import numpy as np

if __name__ == "__main__":
    dropout_ratio = 0.5
    x = np.ones(10)

    mask = np.random.rand(10) > dropout_ratio
    y = x * mask
    print(y)

    # 학습 시
    mask = np.random.rand(*x.shape) > dropout_ratio
    y = x * mask

    # test 시
    scale = 1 - dropout_ratio
    y = x * scale

    # invertible dropout

    # 학습 시
    scale = 1 / (1 - dropout_ratio)
    mask = np.random.rand(*x.shape) > dropout_ratio
    y = x * mask * scale

    # test 시
    y = x

    """
    <OUTPUT>:
    (gpu) C:\Deep_Learning_Study\DLFromScratch3\additional>python step54_dropout.py
    [1. 1. 0. 0. 0. 0. 0. 1. 1. 0.]
    """
