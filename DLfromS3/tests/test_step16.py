# coding: utf-8

import sys
sys.path.append('..')
import unittest
import numpy as np
from common.baseclass import Variable
from common.funcs import square, add

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