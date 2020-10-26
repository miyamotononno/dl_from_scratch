# coding: utf-8
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
np.random.seed(0)

from dezero import Variable, as_variable
from dezero.models import MLP
import dezero.functions as F


model = MLP((10, 3))

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
# p = F.softmax_simple(y)
# print(y)
# print(p)

loss = F.softmax_cross_entropy_simple(y, t)
loss.backward()
print(loss)