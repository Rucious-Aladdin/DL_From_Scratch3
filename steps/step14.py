import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None: # 초기 Gradient를 1로 설정 (dy/dy = 1)
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # x -> f -> y 형태에 대한 backward 수행
            # gxs를 구하려면 gys(이전 미분)값이 필요(연쇄법칙)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):

                if x.grad is None: # 서로 같은 변수를 이용해 더할경우 미분값 처리
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)
    
    def cleargrad(self): # 같은변수에 다른 계산을 하는 경우 대비
        self.grad = None
        
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __init__(self):
        self.input = None
        self.output = None
    
    def __call__(self, *inputs): 
    # inputs에 list-unpacking을 통한 가변길이 인수 제어
        xs = [x.data for x in inputs] # inputs의 Variable을 끄집어냄
        ys = self.forward(*xs) # inputs에 대한 Forward 연산
        
        if not isinstance(ys, tuple): # 인수가 한개인 경우 튜플로 변환
            ys = (ys, )
        
        outputs = [Variable(as_array(y)) for y in ys] # np.float64로 datatype이 변화하는 것을 방지
        
        for output in outputs: # 각 Variable에 대한 Linked-List 연결
            output.set_creator(self)
        
        # input, output을 저장
        self.inputs = inputs
        self.outputs = outputs

        # input의 길이에 따라 List, 또는 Variable 형태로 리턴
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
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def square(x):
    return Square()(x)

def add(x0, x1):
    return Add()(x0, x1)    

if __name__ == "__main__":
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)