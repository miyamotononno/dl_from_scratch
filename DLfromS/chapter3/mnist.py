import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

from nn_3layers import forward

def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()


def get_data():
  _, (x_test, t_test) = \
      load_mnist(flatten=True, normalize=False, one_hot_label=False)
  return x_test, t_test

def init_network():
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)
  
  return network

def predict(network, x):
  return forward(network, x)

if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
      y = predict(network, x[i])
      p = np.argmax(y) # 最も確率の高い要素のインデックスを抽出
      if p == t[i]:
        accuracy_cnt += 1

    print("Accuracy: "+str(float(accuracy_cnt)/len(x)))