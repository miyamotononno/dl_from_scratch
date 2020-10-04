# coding: utf-8

import sys
sys.path.append('..')
import unittest
import numpy as np
from common.baseclass import Variable
from common.funcs import square

class SquareTest(unittest.TestCase):
  def test_forward(self):
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    self.assertEqual(y.data, expected)

if __name__ == "__main__":
  unittest.main()