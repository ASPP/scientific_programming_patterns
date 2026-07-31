>>> def sum(x, y):
...     return x + y
...
>>> def mult(x, y):
...     return x * y
...
>>> def div(x, y):
...     return x/ y
...
>>> sum(3, 2)
5
>>> x, y = 5, 6
>>>
>>> selection = 'sum'
>>> if selection == 'sum':
...     result = sum(3, 2)
... elif selection == 'mult':
...     result = mult(3, 2)
...
>>> result
5
>>> mult
<function mult at 0x1077705c0>
>>> operation = mult
>>> operation(3, 4)
12
>>> operation = sum
>>> operation(3, 4)
7
>>> def compute(x, y, operation):
...     return operation(x, y)
...
>>> compute(x, y, mult)
30
>>> compute(x, y, sum)
11
>>> compute(x, y, div)
0.8333333333333334
>>> operation_registry = {'sum': sum, 'mult': mult, 'div': div}
>>> compute(x, y, operation_registry[selection])
11
>>> selection
'sum'
>>> selection = 'mult'
>>> operation_registry[selection]
<function mult at 0x1077705c0>
>>> compute(x, y, operation_registry[selection])
30
>>>
