if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import dezero
from PIL import Image
from dezero.models import VGG16

if __name__ == "__main__":
    url = "https://github.com/WegraLee/deep-learning-from-scratch-3/raw/images/zebra.jpg"
    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)
    img.show()
    
    x = VGG16.preprocess(img)
    print(type(x), x.shape)