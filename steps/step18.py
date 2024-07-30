"""
<메모리 절약 모드>
1. 불필요한 미분 결과를 삭제
2. 역전파가 필요없는 정방향 예측모드 제공
"""

import numpy as np
import weakref
import contextlib

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

## Config class -> 역전파 활성모드 또는 아닌 모드.
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
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}는 지원하지 않습니다.")
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 ## Toplolgical Order 를 정의

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 부모 세대 + 1
        """
        함수는 자식을 낳는다 (입력 -> 함수(부모) -> 출력(자식))
        Variable의 Creator의 세대는 부모세대보다 1이 커야한다.
        set_creator함수가 실행될때마다 각 Variable의 세대는 1증가한다.
        """
    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set: # -> 비효율적!
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation) # -> 비효율적!
        add_func(self.creator) # initializing state of funcs list


        while funcs:
            f = funcs.pop()
            # gys -> 전단계 grad
            # gxs backward 후 grad
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
                    add_func(x.creator) # 각 함수가 한번만 추가되도록 변경
            
            ## 각 계산그래프의 말단 노드에 있는 Gradient 값만 유지되도록 설정
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys] # 아직 못외움
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs: # 기억 안났음
                output.set_creator(self)
            self.inputs = inputs

        ##  weakref를 통해 양방향 연결리스트의 한쪽 참조를 약한 참조로 함
        ## 순환 참조를 발생시키지 않아 파이썬의 참조 카운트에 따른 메모리 관리방식에 맞게 작동
        ## 카운트가 0가 되면 객체가 소멸
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0] # 여기도.
    
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
    ## retain_grad = False -> 중간 미분값을 저장하지 않음.
    print("========retain_grad=False=============")
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)
    
    ## 메모리 최적화 -> 정방향 연산시 계산그래프의 연결리스트를 메모리에서 해제
    print("==========Config.enable_backprop===========")
    Config.enable_backprop = True
    print(f"Config.enable_backprop: {Config.enable_backprop}")
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()
    print("y.data.shape", y.data.shape)
    print("x.grad.shape", x.grad.shape)

    Config.enable_backprop = False
    print(f"Config.enable_backprop: {Config.enable_backprop}")
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    print("y.data.shape", y.data.shape)
    print("x.grad", x.grad)
    
    ## with문을 통한 mode전환
    print("===========with문의 활용============")
    print("with using_config('enable_backprop', False):")
    with using_config("enable_backprop", False):
        x = Variable(np.array(2.0))
        y = square(x)
        print("y.data", y.data)
        print("x.grad", x.grad)
    
    print("with no_grad():")
    with no_grad():
        x = Variable(np.array(2.0))
        y = square(x)
        print("y.data", y.data)
        print("x.grad", x.grad)