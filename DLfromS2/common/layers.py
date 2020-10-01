# coding: utf-8

import numpy as np
from common.functions import softmax, cross_entropy_error

class MatMul:
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.x = None

  def forward(self, x):
    W,= self.params
    out = np.dot(x, W)
    self.x = x
    return out

  def backward(self, dout):
    W, = self.params
    dx = np.dot(dout, W.T)
    dW = np.dot(self.x.T, dout)
    self.grads[0][...] = dW # 3点リーダーによるdeep copy
    return dx

class SigmoidWithLoss:
  def __init__(self):
    self.params, self.grads = [], []
    self.loss = None
    self.y = None # sigmoidの出力
    self.t = None # 教師データ

  def forward(self, x, t):
    self.t = t
    self.y = 1 / (1 + np.exp(-x))

    self.loss = cross_entropy_error(np.c_[1 -self.y, self.y], self.t)

    return self.loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]

    dx = (self.y - self.t) * dout / batch_size
    return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

class Embedding:
  """重みパラメータから「単語IDに該当する行(ベクトル)」を抜き出すレイヤ
    (One-Hot表現への変換とMatMulレイヤでの行列の乗算は必要ない)
  """
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.idx = None # 抽出する行のインデックス(単語ID)を配列として格納

  def forward(self, idx):
    W, = self.params
    self.idx = idx
    out = W[idx]
    return out

  def backward(self, dout):
    dW, = self.grads
    dW[...] = 0
    np.add.at(dW, self.idx, dout)
    # 以下と同じ意味
    # for i, word_id in enumerate(self.idx):
    #   dw[word_id] += dout[i]
  
    return None