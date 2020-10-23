# coding: utf-8
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# step45.pyと同じ

# データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# ハイパラの設定
lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

# それか次の1行でまとめることもできる
# optimizer = optimizers.SGD(lr).setup(model)


# 学習の開始
for i in range(max_iter):
  y_pred = model(x)
  loss = F.mean_squared_error(y, y_pred)

  model.cleargrads()
  loss.backward()

  optimizer.update()
  if i % 1000 == 0:
    print(loss)