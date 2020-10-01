# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import pickle

from common.optimizer import SGD
from common.time_layers import TimeEmedding, TimeAffine, TimeSoftmaxWithLoss
from common.trainer import RNNlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from lstm import TimeLSTM


class Rnnlm:
  def __init__(self, vocab_size=10000, word_vec=100, hidden_size=100):
    V,D,H = vocab_size, word_vec, hidden_size
    rn = np.random.randn

    # 重みの初期化
    embed_W = (rn(V, D) / 100).astype('f')
    lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b = np.zeros(4 * H).astype('f')
    affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
    affine_b = np.zeros(V).astype('f')

    # レイヤの生成
    self.layers = [
      TimeEmedding(embed_W),
      TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
      TimeAffine(affine_W, affine_b)
    ]
    self.loss_layer = TimeSoftmaxWithLoss()
    self.lstm_layer = self.layers[1]

    # 全ての重みと勾配をリストにまとめる
    self.params, self.grads = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads

  def predict(self, xs):
    for layer in self.layers:
      xs = layer.forward(xs)
    
    return xs

  def forward(self, xs, ts):
    score = self.predict(xs)
    loss = self.loss_layer.forward(score, ts)
    return loss

  def backward(self, dout=1):
    dout  =self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)

    return dout

  def reset_state(self):
    self.lstm_layer.reset_state()

  def save_params(self, file_name='RNNlm.pkl'):
    with open(file_name, 'wb') as f:
      pickle.dump(self.params, f)

  def load_params(self, file_name='RNNlm.pkl'):
    with open(file_name, 'wb') as f:
      self.params = pickle.load(f)



if __name__ == "__main__":
  batch_size = 20
  word_vec = 100
  hidden_size = 100
  time_size = 35
  lr = 20.0
  max_epoch = 4
  max_grad = 0.25

  # 学習データの読み込み
  corpus, word_to_id, id_to_word = ptb.load_data('train')
  corpus_test, _, _ = ptb.load_data('test')
  vocab_size = len(word_to_id)
  xs = corpus[:-1]
  ts = corpus[1:]

  # モデルの生成
  model = Rnnlm(vocab_size, word_vec, hidden_size)
  optimizer = SGD(lr)
  trainer = RNNlmTrainer(model, optimizer)

  # 勾配クリッピングを適用して学習
  trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
  trainer.plot(ylim=(0, 500))

  # テストデータで評価
  model.reset_state()
  ppl_test = eval_perplexity(model, corpus_test)
  print('test perplexity: ', ppl_test)

  # パラメータの保存
  model.save_params()