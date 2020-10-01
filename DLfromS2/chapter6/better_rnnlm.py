# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import pickle
from common.config import GPU

# GPU = True

from common.optimizer import SGD
from common.base_model import BaseModel
from common.time_layers import TimeAffine, TimeDropout, TimeEmedding, TimeSoftmaxWithLoss
from common.trainer import RNNlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from lstm import TimeLSTM


class BetterRnnlm(BaseModel):
  def __init__(self, vocab_size=10000, word_vec=650, hidden_size=0.5, dropout_ratio=0.5):
    """Rnnの改良版
      LSTMの多層化(2層)
      Dropoutを使用(深さ方向に使用)
      重み共有(EmbeddingレイヤとAffineレイヤで重み共有)
    """
    V,D,H = vocab_size, word_vec, hidden_size
    rn = np.random.randn

    # 重みの初期化
    embed_W = (rn(V, D) / 100).astype('f')
    lstm_Wx1 = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    lstm_Wh1 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b1 = np.zeros(4 * H).astype('f')
    lstm_Wx2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_Wh2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b2 = np.zeros(4 * H).astype('f')
    affine_b = np.zeros(V).astype('f')

    # 3つの改善
    self.layers = [
      TimeEmedding(embed_W),
      TimeDropout(dropout_ratio),
      TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
      TimeDropout(dropout_ratio),
      TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
      TimeDropout(dropout_ratio),
      TimeAffine(embed_W.T, affine_b) # 重み共有
    ]
    self.loss_layer = TimeSoftmaxWithLoss()
    self.lstm_layers = [self.layers[2], self.layers[4]]
    self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

    # 全ての重みと勾配をリストにまとめる
    self.params, self.grads = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads

  def predict(self, xs, train_flg=False):
    for layer in self.drop_layers:
      layer.train_flg = train_flg

    for layer in self.layers:
      xs = layer.forward(xs)    
    return xs

  def forward(self, xs, ts, train_flg=True):
    score = self.predict(xs, train_flg)
    loss = self.loss_layer.forward(score, ts)
    return loss

  def backward(self, dout=1):
    dout  =self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)

    return dout

  def reset_state(self):
    for layer in self.lstm_layers:
      layer.reset_state()



if __name__ == "__main__":
  batch_size = 20
  word_vec = 650
  hidden_size = 650
  time_size = 35
  lr = 20.0
  max_epoch = 40
  max_grad = 0.25
  dropout=0.5

  # 学習データの読み込み
  corpus, word_to_id, id_to_word = ptb.load_data('train')
  corpus_val, _, _ = ptb.load_data('val')
  corpus_test, _, _ = ptb.load_data('test')
  vocab_size = len(word_to_id)
  xs = corpus[:-1]
  ts = corpus[1:]

  # モデルの生成
  model = BetterRnnlm(vocab_size, word_vec, hidden_size, dropout)
  optimizer = SGD(lr)
  trainer = RNNlmTrainer(model, optimizer)

  best_ppl = float('inf')

  for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad)
    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('valid perplexity: ', ppl)

    if best_ppl > ppl:
      best_ppl = ppl
      model.save_params()
    else:
      lr /= 4.0
      optimizer.lr = lr

    model.reset_state()

    print('-' * 50)