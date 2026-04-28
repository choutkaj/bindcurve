# BindCurve architecture

This document describes the intended architecture for the rewritten `bindcurve` package.

## Scope

The refactor focuses only on dose-response and binding-curve fitting through `DoseResponseData`. Other assay types such as ITC, SPR, and kinetic traces are intentionally out of scope for this first rewrite.

## Design principles

- Data are validated.
- Models are registered.
- Fitting is generic.
- Results are structured.
- Plotting is separate.
- Failures are explicit.

## Package organization

```text
bindcurve/
  __init__.py
  datasets/
    __init__.py
    dose_response.py
  modeling/
    __init__.py
    parameters.py
    base.py
    logistic.py
    registry.py
  fitting/
    __init__.py
    settings.py
    calculator.py
  results/
    __init__.py
    core.py
  plotting/
    # future plotting functions
```

Legacy wrappers are intentionally not included in this rewrite.

## Data model

The primary data object is `DoseResponseData`. Internally, data are stored in normalized long-form format:

```text
compound_id | experiment_id | concentration | replicate_id | response | metadata...
```

Required columns:

```text
compound_id | concentration | response
```

Optional columns:

```text
experiment_id | replicate_id | weight | metadata...
```

If `experiment_id` or `replicate_id` is missing, `DoseResponseData` fills sensible defaults.

Nested concepts such as compounds and experiments are views over the canonical table, not independent data owners.

## Units

`bindcurve` is unitless by computation and unit-aware by annotation.

The numerical core accepts floats and does not perform automatic unit conversion. Users must provide all concentration-like values in consistent units.

`DoseResponseData` may store labels such as:

```python
concentration_unit = "uM"
response_unit = "fluorescence"
```

Fitted concentration-like parameters inherit the concentration unit. Response-like parameters inherit the response unit. Dimensionless parameters have no unit.

## Default fitting strategy

The default strategy is `per_experiment`.

For each compound:

1. Split data by independent experiment.
2. Aggregate technical replicates at each concentration.
3. Fit one curve to each independent experiment.
4. Summarize fitted parameters across independent experiments.

For `n = 3` independent experiments and `r = 5` technical replicates, this yields three fitted curves per compound. This avoids pseudoreplication and treats independent-experiment variability as the main source of final uncertainty.

## Alternative fitting strategies

- `per_experiment`: recommended default; one fit per independent experiment.
- `per_compound_summary`: aggregate observations by compound and concentration, then fit one summary curve.
- `pooled`: fit all observations for a compound in one pooled fit.

## Replicate aggregation

Technical replicates are aggregated before fitting in the default strategy.

Initial supported aggregation methods:

```text
mean
median
```

Technical replicate variance may be stored for diagnostics, but final uncertainty for IC50/Kd-like values should usually come from variation between independent experiments.

## Parameter summarization

Concentration-like fitted parameters such as `IC50`, `EC50`, `Kd`, `Kds`, and `Kd3` should be summarized across independent experiments on the log10 scale when all estimates are positive.

The package should report both simple linear summaries and geometric means for concentration-like parameters.

## Models

Models are objects, not strings plus scattered `if model == ...` branches.

Each model should know:

- its name;
- its parameter specifications;
- which parameters are concentration-like;
- which parameters are response-like;
- how to evaluate `y = f(x, parameters)`;
- how to generate initial guesses;
- how to build lmfit parameters;
- how to compute residuals.

The base class is `BaseDoseResponseModel`.

The first implemented model is `IC50Model`.

## Model registry

A registry maps string names to model objects:

```python
model = get_model("ic50")
```

This supports a simple public API:

```python
results = bindcurve.fit(data, model="ic50")
```

without hard-coding model-specific logic into the calculator.

## Calculator

`FitCalculator` is the generic fitting coordinator. It should not know model equations. It coordinates:

- data selection;
- replicate aggregation;
- experiment splitting;
- initial parameter generation;
- lmfit minimization;
- result construction;
- summary calculation;
- error handling.

Typical usage:

```python
import bindcurve as bc

calculator = bc.FitCalculator(
    model=bc.IC50Model(),
    settings=bc.FitSettings(strategy="per_experiment"),
)

results = calculator.fit(data, fixed={"ymin": 0.0, "ymax": 100.0})
```

A convenience wrapper also exists:

```python
results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})
```

## Settings

Large function signatures are avoided. Fitting behavior is controlled through settings objects.

Example:

```python
FitSettings(
    strategy="per_experiment",
    replicate_aggregation="mean",
    weighting="none",
    lmfit_method="leastsq",
    errors="raise",
)
```

Do not use `False` as a sentinel for optional numeric values. Use `None` or explicit mappings.

Bad:

```python
fix_ymin=False
```

Good:

```python
fixed={"ymin": 0.0}
```

## Results

Results are structured objects:

```text
ParameterEstimate
FitMetrics
FitResult
FitResults
```

`FitResult` represents one fitted curve. In the default strategy, that means one compound in one independent experiment.

`FitResults` is a collection of `FitResult` objects and provides dataframe exports for individual fits and compound-level summaries.

## Failure handling

Silent failures are not allowed.

Default behavior:

```python
FitSettings(errors="raise")
```

Optional behavior:

```python
FitSettings(errors="collect")
```

In collect mode, failed fits are returned as `FitResult` objects with `success = False` and a failure message.

## Plotting

Plotting should be separate from fitting. Future plotting functions should consume `DoseResponseData + FitResults` and should not redo fitting logic internally.

Potential future functions:

```python
plot_curves(data, results)
plot_grid(data, results)
plot_residuals(results)
```

## First PR target

The first PR implements only the foundation:

- `DoseResponseData`
- `CompoundData`
- `ParameterSpec`
- `BaseDoseResponseModel`
- `IC50Model`
- `FitSettings`
- `FitCalculator`
- `ParameterEstimate`
- `FitMetrics`
- `FitResult`
- `FitResults`
- tests for synthetic IC50 fitting

Additional models should be added after the skeleton is stable and tested.
