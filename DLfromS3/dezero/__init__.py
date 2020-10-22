
is_simple_core = False # step32以前ではTrueにする

if is_simple_core:
  from dezero.core_simple import (Variable, Function, using_config, no_grad, Config,
                                  as_array, as_variable, setup_variable, add)

else:
  from dezero.core import (Variable, Function, using_config, no_grad, Config,
                           as_array, as_variable, setup_variable, Parameter)
  import dezero.functions
  import dezero.utils
  from dezero.layers import Layer
  from dezero.models import Model
setup_variable()