# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
import pickle
from chapter2.section3 import most_similar
from common.util import analogy

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
  params = pickle.load(f)
  word_vecs = params['word_vecs']
  word_to_id = params['word_to_id']
  id_to_word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
  most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)


