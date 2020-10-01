import numpy as np

class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr

  def update(self, params, grads):
    for key in params.keys():
      params[key] -= self.lr * grads[key]


class Momentum:
  def __init__(self, lr=0.01, momentum=0.9):
    self.lr = lr
    self.momentum = momentum
    self.v = None # 速度

  def update(self, params, grads):
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)

    for key in params.keys():
      self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] # self.momentum*self.v[key]は空気抵抗や摩擦をイメージ
      params[key] += self.v[key]

class AdaGrad:
  """
    学習係数を減衰する。
    パラメータ毎に計算。
    パラメータ要素の中で大きく更新された要素は、学習係数が小さくなる。
  """

  def __init__(self, lr=0.01):
    self.lr = lr
    self.h = None # これまで経験した勾配の値を二乗和として保持する

  def update(self, params, grads):
    if self.h is None:
      self.h = {}
      for key, val in params.items():
        self.h[key] = np.zeros_like(val)

    for key in params.keys():
      self.h[key] += grads[key]**2
      params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))


class RMSprop:
  """
    過去の全ての勾配を均一に加算するのでなく、過去の勾配を徐々に忘れて、
    新しい勾配の情報が反映されるように加算する。専門的には「指数移動平均と呼ぶ」
  """
  def __init__(self, lr=0.01, decay_rate=0.09):
    self.lr = lr
    self.decay_rate = decay_rate
    self.h = None

  def update(self, params, grads):
    if self.h is None:
      self.h = {}
      for key, val in params.items():
        self.h[key] = np.zeros_like(val)

    for key in params.keys():
      self.h[key] *= self.decay_rate
      self.h[key] += (1-self.decay_rate) * grads[key]**2
      params[key] -=  self.lr *grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
  """MomentumとRMSPropsの組み合わせ"""
  def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
    self.lr =lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.iter = 0
    self.m = None
    self.v = None

  def update(self, params, grads):
    if self.m is None:
      self.m, self.v = {}, {}
      for key, val in params.items():
        self.m[key] = np.zeros_like(val)
        self.v[key] = np.zeros_like(val)

    self.iter += 1
    lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1 ** self.iter)

    for key in params.keys():
      self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key]) 
      self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

      params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key])+1e-7)