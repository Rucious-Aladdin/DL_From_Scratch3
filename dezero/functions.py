import numpy as np
from dezero.core import Function, as_variable
from dezero import utils
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * - sin(x)
        return gx
    
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        # y를 가지고 올 때에는 self.outputs[0] (tuple),  + () (약한 참조) 형식으로 들고 와야함.
        y = self.outputs[0]() 
        gx = gy * (1 - y * y)
        return gx

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape # <- target shape for y
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    # NOTE: backward 함수는 forward를 함수를 재사용하여 구현.
    def backward(self, gy):
        # NOTE: 원래 형상으로 되돌리기 위해 저장해둔 self.x_shape를 인수로 넘김.
        return reshape(gy, self.x_shape)
    
class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        gx = transpose(gy)
        return gx

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

class Sigmoid(Function):
    def forward(self, x):
        y = np.tanh(x * .5) * .5 + .5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y
    
    def backward(self, gys):
        x, = self.inputs
        gy, = gys
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask 
        return gx # clip된 값의 미분은 0임

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

def clip(x, x_min, x_max):
    x = as_variable(x)
    return Clip(x_min, x_max)(x)

def softmax_cross_entropy_simple(x, t):
    """
    x.shape = (N, C)
    t.shape = (N, )
    t is index of correct label
    """
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0) # log(0)을 방지하기 위해 p의 최솟값을 1e-15이상으로 설정
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

def softmax(x, axis=1):
    return Softmax(axis)(x)

def get_item(x, slices):
    return GetItem(slices)(x)

def sigmoid(x):
    return Sigmoid()(x)

def linear(x, W, b=None):
    return Linear()(x, W, b)

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def matmul(x, W):
    return MatMul()(x, W)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def transpose(x):
    return Transpose()(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)