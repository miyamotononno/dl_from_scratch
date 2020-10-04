# coding: utf-8
import numpy as np
from common.baseclass import Function

class Square(Function):
  def forward(self, x):
    y = x ** 2
    return y

  def backward(self, gy):
    x = self.inputs[0].data
    gx = 2 * x * gy
    return gx

class Exp(Function):
  def forward(self, x):
    y = np.exp(x)
    return y

  def backward(self, gy):
    x = self.inputs[0].data
    gx = np.exp(x) * gy
    return gx

class Add(Function):
  def forward(self, x0, x1):
    return x0 + x1

  def backward(self, gy):
    return gy, gy