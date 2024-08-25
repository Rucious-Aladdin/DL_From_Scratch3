if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dezero
import numpy as np
from dezero.models import VGG16
from PIL import Image
import dezero.cuda
from dezero import Variable

if __name__ == "__main__":
    url = "https://github.com/WegraLee/deep-learning-from-scratch-3/raw/images/zebra.jpg"
    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)
    x = VGG16.preprocess(img)
    x = x[np.newaxis]
    x = Variable(x)
    model = VGG16(pretrained=True)

    if dezero.cuda.gpu_enable:
        model.to_gpu()
        x.to_gpu()
    with dezero.test_mode():
        y = model(x)

    predict_id = np.argmax(y.data)

    model.plot(x, to_file="./viz_images/step58_vgg.pdf")
    labels = dezero.datasets.ImageNet.labels()
    print(labels[predict_id.item()])
    """
    PS C:\Deep_Learning_Study\DLFromScratch3\steps> python .\step58.py
    340
    zebra
    """