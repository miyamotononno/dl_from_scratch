# coding: utf-8

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import test_mode
import dezero.functions as F
import numpy as np

x = np.ones(5)
print(x)

# 学習時
y = F.dropout(x)
print(y)

# テスト時
with test_mode():
  y = F.dropout(x)
  print(y)