# Models

Built-in models are retrieved by name with {py:func}`~bindcurve.get_model`;
[Models](../user-guide/models.md) in the user guide lists them. A custom
model subclasses {py:class}`~bindcurve.Model`, or
{py:class}`~bindcurve.BindingModel` when it computes equilibrium species.

```{eval-rst}
.. autofunction:: bindcurve.get_model

.. autoclass:: bindcurve.Model
   :members: evaluate, guess, parameter

.. autoclass:: bindcurve.BindingModel
   :members: species

.. autoclass:: bindcurve.Parameter
```
