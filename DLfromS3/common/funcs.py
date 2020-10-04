# coding: utf-8
import numpy as np
from common.classes import Square, Exp, Add

def square(x):
  return Square()(x)

def exp(x):
  return Exp()(x)

def add(x0, x1):
  return Add()(x0, x1)