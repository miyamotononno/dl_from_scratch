# coding: utf-8
import numpy as np

def as_array(x):
  return np.array(x) if np.isscalar(x) else x