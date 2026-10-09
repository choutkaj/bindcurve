# Models

Built-in models are retrieved by name with `get_model`.
Custom models subclass `Model`, or `BindingModel` for equilibrium species.

```{eval-rst}
.. autofunction:: bindcurve.get_model

.. autoclass:: bindcurve.Model
   :members: evaluate, guess, parameter

.. autoclass:: bindcurve.BindingModel
   :members: species

.. autoclass:: bindcurve.Parameter
```
