"""
<Variable 편의성 개선>
1. @property decorator 활용 -> numpy.ndarray 처럼 취급
2. __len__, __repr__ 재정의 -> python의 주요 메서드의 오버라이딩 수행
"""

import numpy as np
import weakref
import contextlib

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

class Variable:
    def __init__(self, data, name=None): # 변수 이름 추가
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}는 지원하지 않습니다.")
        
        self.data = data
        self.name = name # 변수 이름 추가
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)
        add_func(self.creator)


        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        else:
            p = str(self.data).replace("\n", "\n" + " " * 9)
            return  "variable(" + p + ")"

class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs

        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy
    
def square(x):
    return Square()(x)

def add(x0, x1):
    return Add()(x0, x1)    

if __name__ == "__main__":
    print("=" * 30)
    x = Variable(np.array([1, 2, 3]))
    print(f"x:\n{x}")
    print("len(x)", len(x))
    
    print("=" * 30)
    x = Variable(None)
    print(f"x:\n{x}")
    
    print("=" * 30)
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(f"x:\n{x}")
    print("len(x)", len(x))