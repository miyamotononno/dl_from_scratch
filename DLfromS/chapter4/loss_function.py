import numpy as np


def simple_mean_squared_error(y, t): # 2乘誤差
  return 0.5 * np.sum((y-t)**2)

def simple_cross_entropy_error(y, t):
  delta = 1e-7
  return - np.sum(t*np.log(y+delta))

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = y.reshape(1, t.size)
    y = y.reshape(1, y.size)

  # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
  if t.size == y.size:
      t = t.argmax(axis=1)
            
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

