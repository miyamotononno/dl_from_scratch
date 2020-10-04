# coding: utf-8

import contextlib

class Config:
  enable_backprop = True # 推論の時はFalseにすることでメモリを節約

@contextlib.contextmanager # withブロックのスコープが入る時に前処理, スコープを抜けると後処理が呼ばれる
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config('enable_backprop', False)
  