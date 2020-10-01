# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from ppmi import ppmi
from section3 import preprocess, create_to_matrix, most_similar
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-pccurrence')
C = create_to_matrix(corpus, vocab_size, window_size)
W = ppmi(C, verbose=True)

print('calculating SVD')
try:
  # truncatted SVD(fast!)
  from sklearn.utils.extmath import randomized_svd
  print('truncatted SVD(fast!)')
  U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, randomized_state=None)
except ImportError:
  # SVD(slow)
  print('SVD(slow)')
  U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
  most_similar(query, word_to_id, id_to_word, word_vecs, top=5)