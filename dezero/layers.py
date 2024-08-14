if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import os
import numpy as np
import weakref
from dezero.core import Parameter
import dezero.functions as F

class Layer:
    def __init__(self) -> None:
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params() 
                # self._params의 객체를 재귀적으로 반환하는 generator
                # DFS 방식
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1] # NOTE: shape[0] is batch_dimension
            self._init_W()
        y = F.linear(x, self.W, self.b)
        return y

if __name__ == "__main__":

    from core import Variable
    layer = Layer()

    layer.p1 = Parameter(np.array(1))
    layer.p2 = Parameter(np.array(2))
    layer.p3 = Variable(np.array(3)) ## Variable 객체는 _params set에 포함되지 않음
    layer.p4 = "test"

    print(layer._params)
    print("-" * 20)

    for name in layer._params:
        print(name, layer.__dict__[name])