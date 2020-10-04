# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import unittest
from common.config import Config, no_grad
from common.baseclass import Variable
from common.funcs import square, add

class Step18Test(unittest.TestCase):
  def test_config(self):
    # 学習時
    self.assertTrue(Config.enable_backprop)
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()
    with no_grad():
      x = Variable(np.array(2.0))
      y = square(x)
      self.assertFalse(Config.enable_backprop)
    
    self.assertTrue(Config.enable_backprop)

if __name__ == "__main__":
  unittest.main()