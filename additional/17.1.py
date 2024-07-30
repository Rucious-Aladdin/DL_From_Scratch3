import weakref
import numpy as np

a = np.array([1, 2, 3])
b = weakref.ref(a) # 약한 참조는 객체를 참조할때 참조 카운트를 증가 시키지 않음!
print(b)
a = None
print(b)

"""
<TERMINAL OUPUT>:

<weakref at 0x0000020E26900900; to 'numpy.ndarray' at 0x0000020E50CB9AD0>
<weakref at 0x0000020E26900900; dead>

"""