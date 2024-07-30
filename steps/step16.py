import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

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

    def backward(self):
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
            gys = [output.grad for output in f.outputs]
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
        
        self.generation = max([x.generation for x in inputs])
        for output in outputs: # 기억 안났음
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs 
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
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def square(x):
    return Square()(x)

def add(x0, x1):
    return Add()(x0, x1)    

if __name__ == "__main__":
    generations = [2, 0, 1, 4, 2]
    funcs = []

    for g in generations:
        f = Function()
        f.generation = g
        funcs.append(f)
    
    print([f.generation for f in funcs])

    funcs.sort(key=lambda x : x.generation)
    print(f.generation for f in funcs)

    f = funcs.pop()
    print(f.generation)

    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    
    print("y.data", y.data)
    print("x.grad", x.grad)