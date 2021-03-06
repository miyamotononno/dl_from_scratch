# coding: utf-8
import numpy as np
import weakref
import contextlib
import dezero

class Config:
  enable_backprop = True # 推論の時はFalseにすることでメモリを節約
  train = True

@contextlib.contextmanager # withブロックのスコープが入る時に前処理, スコープを抜けると後処理が呼ばれる
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def test_mode():
  return using_config('train', False)

def no_grad():
  return using_config('enable_backprop', False)


class Variable:
  __array_priority__ = 200 # 演算子の優先度を高くする

  def __init__(self, data, name=None):
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('{} is not supported'.format(type(data)))

    self.data = data
    self.grad = None # dataの微分値
    self.creator = None # この変数の生成元の関数
    self.generation = 0 # 何層目に位置する変数か
    self.name = name

  def reshape(self, *shape):
    if len(shape)==1 and isinstance(shape[0], (tuple, list)):
      shape=shape[0]
    return dezero.functions.reshape(self, shape)

  def transpose(self):
    return dezero.functions.transpose(self)

  def sum(self, axis=None, keepdims=False):
    return dezero.functions.sum(self, axis, keepdims)
  
  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1 

  def cleargrad(self):
    self.grad = None

  def backward(self, retain_grad=False, create_graph=False):
    if self.grad is None:
      self.grad = Variable(np.ones_like(self.data))

    funcs = []
    seen_set = set()

    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)

    add_func(self.creator)

    while funcs:
      f = funcs.pop()

      gys = [output().grad for output in f.outputs] #outputはweakrefによって生成されたのでoutput()

      with using_config('enable_backprop', create_graph):
        gxs = f.backward(*gys) # メインのbackward
        if not isinstance(gxs, tuple):
          gxs = (gxs, )

        for x, gx in zip(f.inputs, gxs):
          if x.grad is None:
            x.grad = gx
          else:
            x.grad = x.grad + gx # この計算も対象
          
          if x.creator is not None:
            add_func(x.creator)

      if not retain_grad:
        for y in f.outputs:
          y().grad = None # 参照カウントが0となり微分のデータをメモリから削除

  @property # これをつけることでインスタンス変数としてアクセスできる
  def shape(self):
    return self.data.shape

  @property
  def ndim(self):
    """次元数"""
    return self.data.ndim

  @property
  def size(self):
    """要素の数"""
    return self.data.size

  @property
  def dtype(self):
    """データの型"""
    return self.data.dtype

  @property
  def T(self):
    return dezero.functions.transpose(self)

  def __len__(self): # Variableクラスにlenを使用できるようにする
    return len(self.data)

  def __repr__(self): # print関数で出力される文字列をカスタマイズする
    if self.data is None:
      return 'variable(None)'
    p = str(self.data).replace('\n', '\n'+' '* 9)
    return 'variable(' + p +')'


def as_array(x):
  return np.array(x) if np.isscalar(x) else x

def as_variable(obj):
  return obj if isinstance(obj, Variable) else Variable(obj)


class Function:
  def __call__(self, *inputs):
    inputs = [as_variable(x) for x in inputs]
    xs = [x.data for x in inputs]
    ys = self.forward(*xs) # アンパッキング
    if not isinstance(ys, tuple):
      ys = (ys, ) 
    outputs = [Variable(as_array(y)) for y in ys]

    if Config.enable_backprop: # 推論の時は不要
      self.generation = max([x.generation for x in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs
      self.outputs = [weakref.ref(output) for output in outputs] # 弱参照することでメモリ解放を早める(step17)

    return outputs if len(outputs) > 1 else outputs[0]

  def forward(self, xs):
    raise NotImplementedError

  def backward(self, ys):
    raise NotImplementedError

class Add(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    y = x0 + x1
    return y

  def backward(self, gy):
    gx0, gx1 = gy, gy
    if self.x0_shape != self.x1_shape:  # for broadcaset
      gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
      gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
    return gx0, gx1

def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0, x1)

class Mul(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0*x1

  def backward(self, gy):
    x0, x1 = self.inputs
    gx0=gy*x1
    gx1=gy*x0
    if x0.shape != x1.shape:
      gx0 = dezero.functions.sum_to(gx0, x0.shape)
      gx1 = dezero.functions.sum_to(gx1, x1.shape)
    return gx0, gx1

def mul(x0, x1):
  x1 = as_array(x1)
  return Mul()(x0, x1)

class Neg(Function):
  def forward(self, x):
    return -x
  
  def backward(self, gy):
    return -gy

def neg(x):
  return Neg()(x)

class Sub(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0-x1

  def backward(self, gy):
    gx0 = gy
    gx1 = -gy
    if self.x0_shape != self.x1_shape:
      gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
      gx1 = dezero.functions.sum_to(gx1, self.x1_shape) 
    return gx0, gx1

def sub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x0, x1)

def rsub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x1, x0) #x1, x0を逆に入れ替え

class Div(Function):
  def forward(self, x0, x1):
    return x0/x1

  def backward(self, gy):
    x0, x1 = self.inputs
    gx0 = gy / x1
    gx1 = gy * (-x0 / x1 ** 2)
    if x0.shape != x1.shape:  # for broadcast
        gx0 = dezero.functions.sum_to(gx0, x0.shape)
        gx1 = dezero.functions.sum_to(gx1, x1.shape)
    return gx0, gx1

def div(x0, x1):
  x1 = as_array(x1)
  return Div()(x0, x1)

def rdiv(x0, x1):
  x1 = as_array(x1)
  return Div()(x1, x0) #x1, x0を逆に入れ替え

class Pow(Function):
  def __init__(self, c):
    self.c = c

  def forward(self, x):
    return x**self.c

  def backward(self, gy):
    x, = self.inputs
    c = self.c 
    gx = c * x ** (c-1)*gy
    return gx

def dezero_pow(x, c):
  return Pow(c)(x)

class Parameter(Variable):
  pass

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

def get_item(x, slices):
  return GetItem(slices)(x)

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

def setup_variable():
  """以下、演算子のオーバーロードを行う。詳しくはstep20-22参照"""
  Variable.__add__ = add
  Variable.__radd__ = add
  Variable.__mul__ = mul
  Variable.__rmul__ = mul
  Variable.__neg__ = neg
  Variable.__sub__ = sub
  Variable.__rsub__ = rsub
  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv
  Variable.__pow__ = dezero_pow
  Variable.__getitem__ = get_item

  Variable.max = dezero.functions.max
  Variable.min = dezero.functions.min