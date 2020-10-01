import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
  h = 1e-4
  return (f(x+h)-f(x-h))/(2*h)

def func1(x):
  return 0.01*x**2+0.1*x

if __name__ == "__main__":
  y = numerical_diff(func1, np.array([3.0, 4.0]))
  print(y)
