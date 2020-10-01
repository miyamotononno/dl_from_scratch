# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from common.base_model import BaseModel
from common.optimizer import Adam
from common.time_layers import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss
from common.trainer import Trainer
from common.util import eval_seq2seq
from chapter6.lstm import TimeLSTM
from dataset import sequence
from seq2seq import Seq2seq, Encoder

class PeekyDecoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    V, D, H = vocab_size, wordvec_size, hidden_size
    rn = np.random.randn

    embed_W = (rn(V, D) / 100).astype('f')
    lstm_Wx = (rn(D+H, 4*H) / np.sqrt(D)).astype('f')
    lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b = np.zeros(4*H).astype('f')
    affine_W = (rn(H+H, V) / np.sqrt(H)).astype('f')
    affine_b = np.zeros(V).astype('f')

    self.embed = TimeEmbedding(embed_W)
    self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
    self.affine = TimeAffine(affine_W, affine_b)

    self.params, self.grads = [], []
    for layer in (self.embed, self.lstm, self.affine):
      self.params += layer.params
      self.grads += layer.grads
    self.cache = None


  def forward(self, xs, h):

    N, T = xs.shape
    N, H = h.shape

    self.lstm.set_state(h)

    out = self.embed.forward(xs)
    hs = np.repeat(h, T, axis=0).reshape(N, T, H)
    out = np.concatenate((hs, out), axis=2)

    out = self.lstm.forward(out)
    out = np.concatenate((hs, out), axis=2)

    score = self.affine.forward(out)
    self.cache=H
    return score

  def backward(self, dscore):
    H = self.cache

    dout = self.affine.backward(dscore)
    dout, dhs0 = dout[:, :,H:], dout[:,:,:H]
    dout = self.lstm.backward(dout)
    dembed, dhs1 = dout[:, :,H:], dout[:,:,:H]
    dout = self.embed.backward(dembed)
    dhs = dhs0+dhs1
    dh = self.lstm.dh + np.sum(dhs, axis=1)
    return dh

  def generate(self, h, start_id, sample_size):
    sampled = []
    char_id = start_id
    self.lstm.set_state(h)

    H = h.shape[1] 
    peeky_h = h.reshape(1, 1, H)
    for _ in range(sample_size):
      x = np.array([char_id]).reshape((1, 1))
      out = self.embed.forward(x)

      out = np.concatenate((peeky_h, out), axis=2)
      out = self.lstm.forward(out)
      out = np.concatenate((peeky_h, out), axis=2)
      score = self.affine.forward(out)

      char_id = np.argmax(score.flatten())
      sampled.append(char_id)

    return sampled

class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

if __name__ == "__main__":
  (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
  x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
  char_to_id, id_to_char = sequence.get_vocab()

  #　ハイパーパラメータの設定
  vocab_size = len(char_to_id)
  wordvec_size = 16
  hidden_size = 128
  batch_size = 128
  max_epoch = 25
  max_grad = 5.0

  model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
  optimizer = Adam()
  trainer = Trainer(model, optimizer)

  acc_list = []
  for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
      question, correct = x_test[[i]], t_test[[i]]
      verbose = i < 10
      correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, True)
    
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%'% (acc*100)) 