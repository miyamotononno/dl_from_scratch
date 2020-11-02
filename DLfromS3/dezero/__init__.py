
is_simple_core = False # step32以前ではTrueにする

if is_simple_core:
  from dezero.core_simple import (Variable, Function, using_config, no_grad, Config,
                                  as_array, as_variable, setup_variable, add)

else:
  from dezero.core import (Variable, Function, using_config,
                           no_grad, Config, as_array, as_variable,
                           setup_variable, Parameter, test_mode)
  import dezero.functions
  import dezero.optimizers
  import dezero.utils
  import dezero.datasets
  from dezero.layers import Layer
  from dezero.models import Model
  from dezero.dataloaders import DataLoader 
  

setup_variable()