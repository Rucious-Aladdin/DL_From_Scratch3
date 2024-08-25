if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import test_mode
import dezero.functions as F

if __name__ == "__main__":
    x = np.ones(5)
    print(x)

    y = F.dropout(x)
    print(y)

    with test_mode():
        y = F.dropout(x)
        print(y)
