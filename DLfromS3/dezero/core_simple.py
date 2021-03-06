# coding: utf-8
import numpy as np
import weakref
import contextlib

class Config:
  enable_backprop = True # 推論の時はFalseにすることでメモリを節約

@contextlib.contextmanager # withブロックのスコープが入る時に前処理, スコープを抜けると後処理が呼ばれる
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

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
        funcs.sort(key=lambda x: x.generation)

    add_func(self.creator)

    while funcs:
      f = funcs.pop()

      gys = [output().grad for output in f.outputs] #outputはweakrefによって生成されたのでoutput()
      gxs = f.backward(*gys)
      if not isinstance(gxs, tuple):
        gxs = (gxs, )

      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad = gx
        else:
          x.grad = x.grad + gx # x.grad+=gxのようなインプレース演算を行うと前の計算結果が上書かれてしまう(詳しくは付録A参照のこと)

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


class Square(Function):
  def forward(self, x):
    y = x ** 2
    return y

  def backward(self, gy):
    x = self.inputs[0].data
    gx = 2 * x * gy
    return gx

def square(x):
  return Square()(x)


class Exp(Function):
  def forward(self, x):
    y = np.exp(x)
    return y

  def backward(self, gy):
    x = self.inputs[0].data
    gx = np.exp(x) * gy
    return gx

def exp(x):
  return Exp()(x)


class Add(Function):
  def forward(self, x0, x1):
    return x0 + x1

  def backward(self, gy):
    return gy, gy

def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0, x1)


class Mul(Function):
  def forward(self, x0, x1):
    return x0*x1

  def backward(self, gy):
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    return gy*x1, gy*x0

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
    return x0-x1

  def backward(self, gy):
    return gy, -gy

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
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    gx0 = gy / x1
    gx1 = gy * (-x0 / x1**2)
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
    x = self.inputs[0].data
    c = self.c 
    gx = c * x ** (c-1)*gy
    return gx

def dezero_pow(x, c):
  return Pow(c)(x)

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