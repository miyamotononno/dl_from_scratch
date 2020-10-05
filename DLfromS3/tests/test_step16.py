# coding: utf-8
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from dezero import Variable, square, add

class Step16Test(unittest.TestCase):
  def test_backward(self):
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    self.assertEqual(y.data, 32.0)
    self.assertEqual(x.grad, 64.0)
    
if __name__ == "__main__":
  unittest.main()