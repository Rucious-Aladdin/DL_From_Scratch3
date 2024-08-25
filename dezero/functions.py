import numpy as np
from dezero.core import Function, as_variable
from dezero import utils, cuda
from dezero import Variable
from dezero import as_array
import dezero
class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * - sin(x)
        return gx
    
class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
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

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

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
        xp = cuda.get_array_module(x)
        self.x_shape = x.shape
        y = xp.broadcast_to(x, self.shape)
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
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * .5) * .5 + .5
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
        xp = dezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
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
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y
    
    def backward(self, gys):
        x, = self.inputs
        gy, = gys
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask 
        return gx # clip된 값의 미분은 0임

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx
    
def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(xp.float32)
        y = x * mask / scale
        return y
    else:
        return x

def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)

def relu(x):
    return ReLU()(x)

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

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
    f = GetItem(slices)
    return f(x)

def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))

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

def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)

mean = average

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

from dezero.functions_conv import conv2d
#from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2d_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import pooling
#from dezero.functions_conv import average_pooling
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow