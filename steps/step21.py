
"""
<Variable Class 개선>:
1. Variable type 객체가 아니더라도 계산이 가능하도록함.
2. float, int와도 함께 사용이 가능하도록 수정
"""
import numpy as np
import weakref
import contextlib

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    else:
        return Variable(obj)

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
    __array_priority__ = 200 # step21 -> Variable type 의 연산자 메서드를 우선 호출
    # step21. 없으면 -> AtributeError: 'numpy.ndarray' object has no attribute 'backward'

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
    
    ## step20. 연산자 오버로드 지원
    def __mul__(self, other):
        return mul(self, other) # step20. self, other -> Variable Instances
    
    def __add__(self, other):
        return add(self, other) # step20. self, other -> Variable Instances
    

class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
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
    
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
def square(x):
    return Square()(x)

def add(x0, x1):
    x1 = as_array(x1) # step21
    return Add()(x0, x1)    

def mul(x0, x1):
    x1 = as_array(x1) # step21
    return Mul()(x0, x1)

Variable.__add__ = add # step21
Variable.__radd__ = add # step21
Variable.__mul__ = mul # step21
Variable.__rmul__ = mul # step21 없으면 -> TypeError: unsupported operand type(s) for *: 'float' and 'Variable'

if __name__ == "__main__":
    print("Using Variable Translation:")
    x = Variable(np.array(2.0))
    y = x + np.array(3.0)
    print(y)
    print()
    
    print("Using as_arry() func. for float/int dtypes:")
    x = Variable(np.array(2.0))
    y = x + 3.0
    y.backward()
    print(f"y: {y}")
    print(f"x.grad: {x.grad}")
    print()
    
    print("Implementing __rmul__():")
    x = Variable(np.array(2.0))
    y = 3.0 * x + 1.0
    y.backward()
    print(f"y: {y}")
    print(f"x.grad: {x.grad}")
    print()

    print("Set Opertor_Priority:")
    x = Variable(np.array([1.0]))
    y = np.array([2.0]) + x
    y.backward()
    print(f"y: {y}")
    print(f"x.grad: {x.grad}")
    print()