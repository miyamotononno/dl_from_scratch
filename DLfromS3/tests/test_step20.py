# coding: utf-8
if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from dezero import Variable

class Step18Test(unittest.TestCase):
  def test_overload(self):
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    y1 = a * b
    self.assertEqual(y1.data, 6.0)
    y2 = 3.0*a+b
    self.assertEqual(y2.data, 11.0)

if __name__ == "__main__":
  unittest.main()