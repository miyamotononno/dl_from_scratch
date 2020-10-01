# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import SGD
from dataset import ptb
from rnn import TimeRNN
from common.time_layers import TimeAffine, TimeEmedding, TimeSoftmaxWithLoss

class SimpleRNNlm:
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    V, D, H = vocab_size, wordvec_size, hidden_size
    rn = np.random.randn

    # 重みの初期化
    embed_W = (rn(V, D) / 100).astype('f')
    rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
    rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
    rnn_b = np.zeros(H).astype('f')
    affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
    affine_b = np.zeros(V).astype('f')

    # レイヤの生成
    self.layers = [
      TimeEmedding(embed_W),
      TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
      TimeAffine(affine_W, affine_b)
    ]
    self.loss_layer = TimeSoftmaxWithLoss()
    self.rnn_layer = self.layers[1]

    # 全ての重みと勾配をリストにまとめる
    self.params, self.grads = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads

  def forward(self, xs, ts):
    for layer in self.layers:
      xs = layer.forward(xs)
    loss = self.loss_layer.forward(xs, ts)
    return loss

  def backward(self, dout=1):
    dout = self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    return dout

  def reset_state(self):
    self.rnn_layer.reset_state()


if __name__ == "__main__":
  batch_size = 10
  wordvec_size = 100
  hidden_size = 100
  time_size = 5 # Truncated BPTTの展開する時間サイズ
  lr = 0.1
  max_epoch = 100

  # 学習データの読み込み(データセットを小さくする)
  corpus, word_to_id, id_to_word = ptb.load_data('train')
  corpus_size = 1000
  corpus = corpus[:corpus_size]
  vocab_size = int(max(corpus)+1)

  xs = corpus[:-1] # 入力
  ts = corpus[1:] # 出力(教師ラベル)
  data_size = len(xs)
  print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

  # 学習時に使用する変数
  max_iters = data_size // (batch_size * time_size)
  time_idx = 0
  total_loss = 0
  loss_count = 0
  ppl_list = []

  # モデルの生成
  model = SimpleRNNlm(vocab_size, wordvec_size, hidden_size)
  optimizer = SGD(lr)

  # ミニバッチの各サンプルの読み込み開始位置を計算
  jump = (corpus_size - 1) // batch_size
  offsets = [i * jump for i in range(batch_size)]

  for epoch in range(max_epoch):
    for itr in range(max_iters):
      # ミニバッチの取得
      batch_x = np.empty((batch_size, time_size), dtype='i')
      batch_t = np.empty((batch_size, time_size), dtype='i')
      for t in range(time_size):
        for i, offset in enumerate(offsets):
          batch_x[i, t] = xs[(offset + time_idx) % data_size]
          batch_t[i, t] = ts[(offset + time_idx) % data_size]
        time_idx += 1
      
      # 勾配を求めr、パラメータを更新
      loss = model.forward(batch_x, batch_t)
      model.backward()
      optimizer.update(model.params, model.grads)
      total_loss += loss
      loss_count += 1

    # エポックごとにperplexityの評価
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0 