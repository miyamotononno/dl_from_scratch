import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from chapter3.activation_function import sigmoid

def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def softmax(x):
  # x -= np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
  # return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

  x -= np.max(x)
  y = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True) # axis=-1, keepdims=Trueがないとエラー

  # 以下はうまくいかないので注意！！
  # c = np.max(x)
  # exp_a = np.exp(x-c)
  # sum_exp_a = np.sum(exp_a) # overflow対策
  # y = exp_a / sum_exp_a
  return y

def forward(network, x):
  W1, W2, W3 = network['W1'],network['W2'],network['W3']
  b1, b2, b3 = network['b1'],network['b2'],network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3) # 他クラス分類を想定, ２クラス分類のときはシグモイド, 回帰問題の場合は恒等関数
  
  return y

if __name__ == "__main__":
  network = init_network()
  x = np.array([1.0, 0.5])
  y = forward(network, x)
  print(y)