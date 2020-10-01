import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from collections import OrderedDict
from chapter4.gradient import numerical_gradient
from layer_activation import Affine, SoftmaxWithLoss, Relu

class TwoLayerNet:
  
  def __init__(self, input_size, hidden_size, output_size,
               weight_init_std=0.01):
    # 重みの初期化
    self.params = {}
    self.params['W1'] = weight_init_std * \
                        np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * \
                        np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    # レイヤの生成
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    self.lastLayer = SoftmaxWithLoss()
  
  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)

    return x

  # x: 入力データ, t: 教師データ
  def loss(self, x, t):
    y = self.predict(x)

    return self.lastLayer.forward(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1: t = np.argmax(t, axis=1)

    accuracy = np.sum(y==t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    """gradient関数がうまくいくかどうかのチェックに使う"""
    loss_W = lambda W: self.loss(x,t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads

  def gradient(self, x, t):
    # forward
    self.loss(x,t)

    #backward
    dout = 1
    dout = self.lastLayer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout) 

    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db

    return grads

if __name__ == "__main__":

    (x_train, t_train), (x_test, t_test) = \
      load_mnist(normalize=True, one_hot_label=True)

    # ハイパーパラメータ
    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
  
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 1エポックあたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)


    network = TwoLayerNet(input_size=784, hidden_size=50, output_size =10)
    x = np.random.randn(100, 784)
    t = np.random.randn(100, 10)
    grads = network.numerical_gradient(x, t)
    for i in range(iter_num):
      # ミニバッチの取得
      batch_mask = np.random.choice(train_size, batch_size)
      x_batch = x_train[batch_mask]
      t_batch = t_train[batch_mask]

      # 勾配の計算
      grad = network.gradient(x_batch, t_batch)

      # パラメータの更新
      for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

      # 学習経過の記録
      loss = network.loss(x_batch, t_batch)
      train_loss_list.append(loss)
      if i % iter_per_epoch==0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | "+ str(train_acc)+", "+ str(test_acc)) 
