import numpy as np
from common.base_model import BaseModel
from common.functions import softmax
from common.layers import Embedding

class TimeAffine(BaseModel):
  def __init__(self, W, b):
    self.params = [W, b]
    self.grads = [np.zeros_like(W), np.zeros_like(b)]
    self.x = None

  def forward(self, x):
    N, T, D = x.shape
    W, b = self.params

    rx = x.reshape(N*T, -1)
    out = np.dot(rx, W) + b
    self.x = x
    return out.reshape(N, T, -1)

  def backward(self, dout):
    x = self.x
    N, T, D = x.shape
    W, b = self.params

    dout = dout.reshape(N*T, -1)
    rx = x.reshape(N*T, -1)

    db = np.sum(dout, axis=0)
    dW = np.dot(rx.T, dout)
    dx = np.dot(dout, W.T)
    dx = dx.reshape(*x.shape)

    self.grads[0][...] = dW
    self.grads[1][...] = db

    return dx


class TimeEmedding(BaseModel):
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.layers = None
    self.W = W

  def forward(self, xs):
    N, T = xs.shape
    V, D = self.W.shape

    out = np.empty((N, T, D), dtype='f')
    self.layers = []

    for t in range(T):
        layer = Embedding(self.W)
        out[:, t, :] = layer.forward(xs[:, t])
        self.layers.append(layer)

    return out

  def backward(self, dout):
    N, T, D = dout.shape

    grad = 0
    for t in range(T):
        layer = self.layers[t]
        layer.backward(dout[:, t, :])
        grad += layer.grads[0]

    self.grads[0][...] = grad
    return None


class TimeSoftmaxWithLoss(BaseModel):
  def __init__(self):
    self.params, self.grads = [], []
    self.cache = None
    self.ignore_label = -1

  def forward(self, xs, ts):
    N, T, V = xs.shape

    if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
        ts = ts.argmax(axis=2)

    mask = (ts != self.ignore_label)

    # バッチ分と時系列分をまとめる（reshape）
    xs = xs.reshape(N * T, V)
    ts = ts.reshape(N * T)
    mask = mask.reshape(N * T)

    ys = softmax(xs)
    ls = np.log(ys[np.arange(N * T), ts])
    ls *= mask  # ignore_labelに該当するデータは損失を0にする
    loss = -np.sum(ls)
    loss /= mask.sum()

    self.cache = (ts, ys, mask, (N, T, V))
    return loss

  def backward(self, dout=1):
    ts, ys, mask, (N, T, V) = self.cache

    dx = ys
    dx[np.arange(N * T), ts] -= 1
    dx *= dout
    dx /= mask.sum()
    dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

    dx = dx.reshape((N, T, V))

    return dx

class TimeDropout(BaseModel):
  def __init__(self, dropout_ratio=0.5):
    self.params, self.grads = [], []
    self.dropout_ratio = dropout_ratio
    self.mask = None
    self.train_flg = True

  def forward(self, xs):
    if self.train_flg:
      flg = np.random.rand(*xs.shape) > self.dropout_ratio
      scale = 1 / (1.0 - self.dropout_ratio)
      self.mask = flg.astype(np.float32) * scale

      return xs * self.mask
    else:
      return xs
  
  def backward(self, dout):
    return dout * self.mask