# coding: utf-8
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from dezero import Variable, square, Config, no_grad

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