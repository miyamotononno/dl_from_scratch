# coding: utf-8
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero
import math
import numpy as np
from dezero import optimizers, DataLoader
import dezero.functions as F
from dezero.models import MLP

# ハイパーパラメータの設定
max_epoch = 5
batch_size = 100
hidden_size = 1000

# 以下step48.pyの改良
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
  sum_loss, sum_acc = 0, 0

  for x, t in train_loader: # 訓練用のミニバッチデータ
    y = model(x)
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t) # 訓練データの認識精度
    model.cleargrads()
    loss.backward()
    optimizer.update()

    sum_loss += float(loss.data) * len(t)
    sum_acc += float(acc.data) * len(t)

  print('epoch: {}'.format(epoch+1))
  print('train loss: {:.4f}, accuracy: {:.4f}'.format(
    sum_loss / len(train_set), sum_acc / len(train_set)))

  sum_loss, sum_acc = 0, 0
  with dezero.no_grad():
    for x, t in test_loader:
      y = model(x)
      loss = F.softmax_cross_entropy(y, t)
      acc = F.accuracy(y, t)
      sum_loss += float(loss.data) * len(t)
      sum_acc += float(acc.data) * len(t)

  print('test loss: {:.4f}, accuracy: {:.4f}'.format(
    sum_loss / len(test_set), sum_acc / len(test_set)))