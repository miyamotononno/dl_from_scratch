
is_simple_core = False # step32以前ではTrueにする

if is_simple_core:
  from dezero.core_simple import (Variable, Function, using_config, no_grad, Config,
                                  as_array, as_variable, setup_variable, add)

else:
  from dezero.core import (Variable, Function, using_config, no_grad, Config,
                           as_array, as_variable, setup_variable)
  import dezero.functions
  import dezero.utils
setup_variable()