import numpy as np
import matplotlib.pyplot as plt
from step01 import Variable
from step02 import Square
from step03 import Exp

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

def f0(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

if __name__ == "__main__":
    f = Square()
    x = Variable(np.array(2.0))
    dy =  numerical_diff(f, x)
    print(dy)

    epss= np.logspace(-20, -1, 100000)
    dys = []
    for eps in epss:
        x = Variable(np.array(2.0))
        dy = numerical_diff(f, x, eps)
        #print(f"{dy:.3f}")
        dys.append(float(numerical_diff(f, x, eps)))
    #print(dys)
    #print(epss.tolist())
    plt.plot(epss.tolist(), dys)
    #plt.ylim(0, 6)
    plt.xscale("log", base=10)
    plt.show()
    # 10 ^ -12 ~ 10 ^ -3 에서 안정적인 수치 미분값 도출.

    x = Variable(np.array(.5))
    dy = numerical_diff(f0, x)
    print(dy)