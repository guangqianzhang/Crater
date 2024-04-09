
import time
from functools import wraps
# 我们有一个名为fn_timer的函数，它接受一个函数作为参数，并返回一个新的函数function_timer。function_timer是fn_timer的内部函数，
# 它使用了装饰器@wraps来包装传入的函数。
# 当我们调用fn_timer并传入一个函数作为参数时，它会返回function_timer函数。然后，我们可以使用返回的函数来执行传入的函数，并计算其执行时间。
def fn_timer(function):
   @wraps (function)
   def function_timer( *args, **kwargs):
     t0 = time.time()
     result = function( *args, ** kwargs)
     t1 = time.time()
     print ( "Total time running %s: %s seconds" %
         (function.__name__, str (t1 - t0))
         )
     return result
   return function_timer