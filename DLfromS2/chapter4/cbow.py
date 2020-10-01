# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import pickle
from common import config
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import to_cpu, to_gpu
from negative_sampling import NegativeSamplingLoss, Embedding
from chapter3.context_target import create_contexts_target
from dataset import ptb

# GPUで使用する場合は以下をコメントアウト
# config.GPU = True

class CBOW:
  def __init__(self, vocab_size, hidden_size, window_size, corpus):
    V, H = vocab_size, hidden_size

    # 重みの初期化
    W_in = 0.01 * np.random.randn(V, H).astype('f')
    W_out = 0.01 * np.random.randn(V, H).astype('f')

    # レイヤの生成
    self.in_layers = []
    for _ in range(2 * window_size):
      layer = Embedding(W_in) 
      self.in_layers.append(layer)
    self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

    # 全ての重みと勾配を配列にまとめる
    layers = self.in_layers + [self.ns_loss]
    self.params, self.grads = [], []
    for layer in layers:
      self.params += layer.params
      self.grads += layer.grads

    # メンバ変数に単語の分散表現を設定
    self.word_vecs = W_in

  def forward(self, contexts, target):
    h = 0
    for i, layer in enumerate(self.in_layers):
      h += layer.forward(contexts[:, i])
    h *= 1 / len(self.in_layers)
    loss = self.ns_loss.forward(h, target)
    return loss

  def backward(self, dout=1):
    dout = self.ns_loss.backward(dout)
    dout *= 1 / len(self.in_layers)
    for layer in self.in_layers:
      layer.backward(dout)
    return None

if __name__ == "__main__":
  window_size = 5
  hidden_size = 100
  batch_size = 100
  max_epoch = 10

  # データの読み込み
  corpus, word_to_id, id_to_word = ptb.load_data('train')
  vocab_size = len(word_to_id)

  contexts, target = create_contexts_target(corpus, window_size)
  if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)
  
  model = CBOW(vocab_size, hidden_size, window_size, corpus)
  optimizer = Adam()
  trainer = Trainer(model, optimizer)

  # 学習開始
  trainer.fit(contexts, target, max_epoch, batch_size)
  trainer.plot()

  # 後ほど使用できるように、必要なデータを保存
  word_vecs = model.word_vecs
  if config.GPU:
    target = to_cpu(word_vecs)
  
  params = {}
  params['word_vecs'] = word_vecs.astype(np.float16)
  params['word_to_id'] = word_to_id
  params['id_to_word'] = id_to_word
  pkl_file = 'cbow_params.pkl'
  with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)