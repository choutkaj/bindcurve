# API reference

The reference below documents the public objects exported from the top-level
`bindcurve` package. Signatures, parameter descriptions, return values, and
class members are generated from the source docstrings.

## Data and fitting

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.DoseResponseData
   bindcurve.FitSettings
   bindcurve.fit
```

## Results and quality

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.FitResults
   bindcurve.FitResult
   bindcurve.FitMetrics
   bindcurve.ParameterEstimate
   bindcurve.ParameterSummary
   bindcurve.ConcentrationSummary
   bindcurve.DataQualityThresholds
   bindcurve.ResultQualityThresholds
```

## Plotting

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.CurvePoint
   bindcurve.plot_fits
   bindcurve.plot_compounds
   bindcurve.plot_residuals
   bindcurve.plot_asymptotes
   bindcurve.plot_curve_points
```

## Model infrastructure

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.BaseDoseResponseModel
   bindcurve.ModelEvaluation
   bindcurve.ParameterSpec
   bindcurve.get_model
```

## IC50 conversion

```{eval-rst}
.. autosummary::
   :nosignatures:

   bindcurve.IC50ConversionResult
   bindcurve.convert_ic50_to_kd
   bindcurve.cheng_prusoff
   bindcurve.cheng_prusoff_corrected
   bindcurve.coleska
```

## Package metadata

`bindcurve.__version__` contains the installed package version.

```{eval-rst}
.. autodata:: bindcurve.__version__
```

```{toctree}
:hidden:
:maxdepth: 2

data-fitting
results-quality
plotting
modeling
conversion
```
