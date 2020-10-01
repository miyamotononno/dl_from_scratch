# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from section3 import preprocess, create_to_matrix, cos_similarity


def ppmi(C, verbose=False, eps=1e-8):
  """正の相互情報量(positive Pointwise Mutual Information)"""
  M = np.zeros_like(C, dtype=np.float32)
  N = np.sum(C)
  S = np.sum(C, axis=0)
  total = C.shape[0] * C.shape[1]
  cnt = 0

  for i in range(C.shape[0]):
    for j in range(C.shape[1]):
      pmi = np.log2(C[i, j]*N/(S[i]*S[j])+eps)
      M[i, j] = max(0, pmi)

      if verbose:
        cnt+=1

      if cnt % (total//100 + 1) == 0:
        print('%.1f%% done' % (100*cnt/total))
  
  return M


if __name__ == "__main__":
  text = 'You say goodbye and I say hello.'
  corpus, word_to_id, id_to_word = preprocess(text)
  vocab_size = len(word_to_id)
  C = create_to_matrix(corpus, vocab_size)
  W = ppmi(C, verbose=True)

  np.set_printoptions(precision=3) #有効数字3桁
  print('covariance matrix')
  print(C)
  print('-'*50)
  print('PPMI')
  print(W)