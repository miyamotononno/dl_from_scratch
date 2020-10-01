# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import pickle

from common.functions import softmax
from chapter6.rnnlm import Rnnlm
from dataset import ptb

class RnnlmGen(Rnnlm):
  def generate(self, start_id, skip_ids=None, sample_size=100):
    word_ids = [start_id]

    x = start_id
    while len(word_ids) < sample_size:
      x = np.array(x).reshape(1, 1)
      score = self.predict(x)
      p = softmax(score.flatten())

      sampled = np.random.choice(len(p), size=1, p=p)
      if (skip_ids is None) or (sampled not in skip_ids):
        x = sampled
        word_ids.append(int(x))

    return word_ids

if __name__ == "__main__":
  corpus, word_to_id, id_to_word = ptb.load_data('train')
  vocab_size = len(word_to_id)
  corpus_size = len(corpus)

  model = RnnlmGen()
  model.load_params('../chapter6/RNNlm.pkl')

  # start文字とskip文字の設定
  start_word = 'you'
  start_id = word_to_id[start_word]
  skip_words = ['N', '<unk>', '$']
  skip_ids = [word_to_id[w] for w in skip_words]

  # 文章生成
  word_ids = model.generate(start_id, skip_ids)
  txt = ' '.join([id_to_word[i] for i in word_ids])
  txt = txt.replace(' <eos>', '.\n')
  print(txt)