# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from attention_layer import TimeAttention
from chapter6.lstm import TimeLSTM
from chapter7.seq2seq import Encoder, Seq2seq
from common.optimizer import Adam
from common.time_layers import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss
from common.trainer import Trainer
from common.util import eval_seq2seq
from dataset import sequence

class AttentionEncoder(Encoder):
  def forward(self, xs):
    xs = self.embed.forward(xs)
    hs = self.lstm.forward(xs)
    return hs

  def backward(self, dhs):
    dout = self.lstm.backward(dhs)
    dout = self.embed.backward(dout)
    return dout

class AttentionDecoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    V, D, H = vocab_size, wordvec_size, hidden_size
    rn = np.random.randn

    embed_W = (rn(V, D) / 100).astype('f')
    lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b = np.zeros(4*H).astype('f')
    affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
    affine_b = np.zeros(V).astype('f')

    self.embed = TimeEmbedding(embed_W)
    self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
    self.attention = TimeAttention()
    self.affine = TimeAffine(affine_W, affine_b)
    layers = [self.embed, self.lstm, self.attention, self.affine]

    self.params, self.grads = [], []
    for layer in layers:
      self.params += layer.params
      self.grads += layer.grads

  def forward(self, xs, enc_hs):
    h = enc_hs[:, -1]
    self.lstm.set_state(h)

    out = self.embed.forward(xs)
    dec_hs = self.lstm.forward(out)
    c = self.attention.forward(enc_hs, dec_hs)
    out = np.concatenate((c, dec_hs), axis=2)
    score = self.affine.forward(out)

    return score

  def backward(self, dscore):
    dout = self.affine.backward(dscore)
    _, _, H2 = dout.shape
    H = H2 // 2

    dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:]
    denc_hs, ddec_hs1 = self.attention.backward(dc)
    ddec_hs = ddec_hs0 + ddec_hs1
    dout = self.lstm.backward(ddec_hs)
    dh = self.lstm.dh
    denc_hs[:, -1] += dh
    self.embed.backward(dout)

    return denc_hs

  def generate(self, enc_hs, start_id, sample_size):
    sampled = []
    sample_id = start_id
    h = enc_hs[:, -1]
    self.lstm.set_state(h)
    for _ in range(sample_size):
      x = np.array([sample_id]).reshape((1, 1))

      out = self.embed.forward(x)
      dec_hs = self.lstm.forward(out)
      c = self.attention.forward(enc_hs, dec_hs)
      out = np.concatenate((c, dec_hs), axis=2)
      score = self.affine.forward(out)

      sample_id = np.argmax(score.flatten())
      sampled.append(sample_id)

    
    return sampled

class AttentionSeq2seq(Seq2seq):
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    args = vocab_size, wordvec_size, hidden_size
    self.encoder = AttentionEncoder(*args)
    self.decoder = AttentionDecoder(*args)
    self.softmax = TimeSoftmaxWithLoss()

    self.params = self.encoder.params + self.decoder.params
    self.grads = self.encoder.grads + self.decoder.grads 

if __name__ == "__main__":
  (x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
  char_to_id, id_to_char = sequence.get_vocab()

  # 入力文を反転
  x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

  # ハイパーパラメータの設定
  vocab_size = len(char_to_id)
  wordvec_size = 16
  hidden_size = 256
  batch_size = 128
  max_epoch = 4
  max_grad = 5.0

  model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)

  optimizer = Adam()
  trainer = Trainer(model, optimizer)

  acc_list = []
  for epoch in range(max_epoch):
      trainer.fit(x_train, t_train, max_epoch=1,
                  batch_size=batch_size, max_grad=max_grad)

      correct_num = 0
      for i in range(len(x_test)):
          question, correct = x_test[[i]], t_test[[i]]
          verbose = i < 10
          correct_num += eval_seq2seq(model, question, correct,
                                      id_to_char, verbose, is_reverse=True)

      acc = float(correct_num) / len(x_test)
      acc_list.append(acc)
      print('val acc %.3f%%' % (acc * 100))

  model.save_params()
  # グラフの描画
  x = np.arange(len(acc_list))
  plt.plot(x, acc_list, marker='o')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.ylim(-0.05, 1.05)
  plt.show() 