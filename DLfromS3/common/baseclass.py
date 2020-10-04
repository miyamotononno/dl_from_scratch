# coding: utf-8
import numpy as np
from common.utils import *
from common.config import Config
import weakref

class Variable:
  def __init__(self, data):
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('{} is not supported'.format(type(data)))

    self.data = data
    self.grad = None # dataの微分値
    self.creator = None # この変数の生成元の関数
    self.generation = 0 # 何層目に位置する変数か

  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1 

  def clear_glad(self):
    self.glad = None

  def backward(self, retain_grad=False):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

    funcs = []
    seen_set = set()

    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)

    add_func(self.creator)
    while funcs:
      f = funcs.pop()

      gys = [output().grad for output in f.outputs] #outputはweakrefによって生成されたのでoutput()
      gxs = f.backward(*gys)
      if not isinstance(gxs, tuple):
        gxs = (gxs, )
      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad = gx
        else:
          x.grad = x.grad + gx # x.grad+=gxのようなインプレース演算を行うと前の計算結果が上書かれてしまう(詳しくは付録A参照のこと)

        if x.creator is not None:
          add_func(x.creator)

      if not retain_grad:
        for y in f.outputs:
          y().grad = None # 参照カウントが0となり微分のデータをメモリから削除
 

class Function:
  def __call__(self, *inputs):
    xs = [x.data for x in inputs]
    ys = self.forward(*xs) # アンパッキング
    if not isinstance(ys, tuple):
      ys = (ys, ) 
    outputs = [Variable(as_array(y)) for y in ys]

    if Config.enable_backprop: # 推論の時は不要
      self.generation = max([x.generation for x in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs
      self.outputs = [weakref.ref(output) for output in outputs] # 弱参照することでメモリ解放を早める(step17)

    return outputs if len(outputs) > 1 else outputs[0]

  def forward(self, xs):
    raise NotImplementedError

  def backward(self, ys):
    raise NotImplementedError
