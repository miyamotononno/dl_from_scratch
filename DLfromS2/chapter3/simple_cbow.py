# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from chapter2.section3 import preprocess
from common.optimizer import Adam
from common.trainer import Trainer
from common.layers import MatMul, SoftmaxWithLoss
from context_target import create_contexts_target, convert_one_hot

class SimpleCBOW:
  def __init__(self, vocab_size, hidden_size):
    V, H = vocab_size, hidden_size

    # 重みの初期化
    W_in = 0.01 * np.random.randn(V, H).astype('f')
    W_out = 0.01 * np.random.randn(H, V).astype('f')

    # レイヤの生成
    self.in_layer0 = MatMul(W_in)
    self.in_layer1 = MatMul(W_in)
    self.out_layer = MatMul(W_out)
    self.loss_layer = SoftmaxWithLoss()

    # 全ての重みと勾配をリストにまとめる
    layers = [self.in_layer0, self.in_layer1, self.out_layer]
    self.params, self.grads = [], []
    for layer in layers:
      self.params += layer.params
      self.grads += layer.grads

    # メンバ変数に単語の分散表現を設定
    self.word_vec = W_in

  def forward(self, contexts, target): # contextsは3次元であることを想定
    h0 = self.in_layer0.forward(contexts[:, 0])
    h1 = self.in_layer1.forward(contexts[:, 1])
    h = (h0 + h1) * 0.5
    score = self.out_layer.forward(h)
    loss = self.loss_layer.forward(score, target)
    return loss

  def backward(self, dout=1):
    ds = self.loss_layer.backward(dout)
    da = self.out_layer.backward(ds)
    da *= 0.5
    self.in_layer1.backward(da)
    self.in_layer0.backward(da)
    return None

if __name__ == "__main__":
  window_size = 1
  hidden_size = 5
  batch_size = 3
  max_epoch = 1000

  text = 'You say goodbye and I say hello.'
  corpus, word_to_id, id_to_word = preprocess(text)
  vocab_size = len(word_to_id)

  contexts, target = create_contexts_target(corpus, window_size)
  contexts = convert_one_hot(contexts, vocab_size)
  target = convert_one_hot(target, vocab_size)

  model = SimpleCBOW(vocab_size, hidden_size)
  optimizer = Adam()
  trainer = Trainer(model, optimizer)

  trainer.fit(contexts, target, max_epoch, batch_size)
  trainer.plot()