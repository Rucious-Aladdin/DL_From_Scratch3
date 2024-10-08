if "__file__" in globals(): # 일시적으로 사용하는 것이므로 쓸때마다 작성
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import os
import numpy as np
import weakref
from dezero.core import Parameter
import dezero.functions as F
from dezero import cuda
from dezero.utils import pair

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
    
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e: # KeyboardInterrupt 추가함으로써 덜만들어진 파일로 인한 오류 방지
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
       npz = np.load(path)
       params_dict = {}
       self._flatten_params(params_dict)
       for key, param in params_dict.items():
           param.data = npz[key]    

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

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu() 


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

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1] # NOTE: shape[0] is batch_dimension
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        y = F.linear(x, self.W, self.b)
        return y

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=hidden_size, nobias=True)
        self.h = None
    
    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new

class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        ## forget gate
        self.x2f = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=H, nobias=True)

        ## input gate
        self.x2i = Linear(H, in_size=I)
        self.h2i = Linear(H, in_size=H, nobias=True)

        ## output gate
        self.x2o = Linear(H, in_size=I)
        self.h2o = Linear(H, in_size=H, nobias=True)

        ## memory cell(cell_state)
        self.x2c = Linear(H, in_size=I)
        self.h2c = Linear(H, in_size=H, nobias=True)

        self.reset_state()

    def reset_state(self):
        self.h, self.c = None, None
    
    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            c = F.tanh(self.x2c(x))

        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            c = F.tanh(self.x2c(x) + self.h2c(self.h))
        
        if self.c is None:
            c_new = f * c
        else:
            c_new = f * self.c + i * c # next cell state(Adamar product)
        
        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
    
        return h_new # output is new hidden state

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