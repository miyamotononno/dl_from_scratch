import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from chapter3.nn_3layers import softmax
from chapter4.loss_function import cross_entropy_error

def func2(x):
  return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for _ in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr*grad

  return x

class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2, 3) # ガウス分布で初期化

  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

if __name__ == "__main__":
    net = simpleNet()
    print(net.W)