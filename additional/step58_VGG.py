if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero.models import VGG16
import numpy as np

if __name__ == "__main__":
    model = VGG16(pretrained=True)

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    model.plot(x)