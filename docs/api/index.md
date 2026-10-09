# API reference

These are the public objects exported from the top-level `bindcurve` package.

## Data and fitting

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.DoseResponseData
   bindcurve.fit
   bindcurve.FitResults
   bindcurve.FitResult
```

## Plotting

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.plot_fits
   bindcurve.plot_compounds
   bindcurve.plot_residuals
```

## Models

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.get_model
   bindcurve.Model
   bindcurve.BindingModel
   bindcurve.Parameter
```

## IC₅₀ conversion

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.cheng_prusoff
   bindcurve.munson_rodbard
   bindcurve.coleska
```

## Package metadata

```{eval-rst}
.. autodata:: bindcurve.__version__
```

```{toctree}
:hidden:
:maxdepth: 2

data-fitting
plotting
models
conversion
```
