# Bindcurve architecture

This document describes the intended architecture for the rewritten `bindcurve` package.

The rewrite focuses on one domain for now: dose-response and binding-curve fitting. Other assay types such as ITC, SPR, and kinetic time traces are intentionally out of scope for the first refactor. The internal design should remain clean enough that those domains can be added later, but the first implementation should not prematurely generalize beyond `DoseResponseData`.

## Design goals

`bindcurve` should be a serious, lmfit-backed, object-oriented package for fitting dose-response and binding curves.

The core principles are:

- Data are validated.
- Models are registered.
- Fitting is generic.
- Results are structured.
- Plotting is separate.
- Failures are explicit.

The public API should remain simple for notebooks, but the internals should be robust enough for reproducible scientific analysis.

## Scope

The initial rewrite targets only dose-response data:

```text
DoseResponseData
```

The first implementation should include a clean skeleton and one working model, `IC50Model`. Additional models such as logIC50, Hill, direct Kd, competitive Kd, and IC50-to-Kd conversion models should be added in later PRs after the foundation is tested.

## Package organization

The initial skeleton uses non-conflicting module names so it can coexist temporarily with legacy modules during the refactor:

```text
bindcurve/
  __init__.py

  datasets/
    __init__.py
    dose_response.py      # DoseResponseData and CompoundData

  modeling/
    __init__.py
    parameters.py         # ParameterSpec
    base.py               # BaseDoseResponseModel
    logistic.py           # IC50Model
    registry.py           # model registry

  fitting/
    __init__.py
    settings.py           # FitSettings
    calculator.py         # FitCalculator and convenience fit()

  results/
    __init__.py
    core.py               # ParameterEstimate, FitMetrics, FitResult, FitResults

  plotting/
    # future plotting functions

  compat/
    # future deprecated wrappers for the legacy API
```

Once the legacy modules are fully removed, these package names can be revisited. The architectural concepts are more important than the exact directory names.

## Data model

The primary data object is `DoseResponseData`.

Internally, data are stored in normalized long-form format:

```text
compound_id | experiment_id | concentration | replicate_id | response | metadata...
```

Example:

```text
CMPD_1 | exp1 | 0.001 | rep1 | 98.2
CMPD_1 | exp1 | 0.001 | rep2 | 97.9
CMPD_1 | exp2 | 0.001 | rep1 | 95.4
```

This supports:

- multiple compounds,
- multiple independent experiments,
- multiple concentrations,
- multiple technical replicates,
- missing technical replicates,
- future metadata columns.

Wide assay tables may be accepted by loaders, but they should be converted to long-form data immediately.

Nested concepts such as compounds and experiments should be implemented as views over the canonical table, not as the primary storage format.

## Units

`bindcurve` is unitless by computation and unit-aware by annotation.

The numerical core accepts floats and does not perform automatic unit conversion. Users are responsible for providing all concentration-like values in consistent units.

`DoseResponseData` may store labels such as:

```python
concentration_unit = "uM"
response_unit = "fluorescence"
```

Fitted concentration-like parameters inherit the concentration unit. Response-like parameters inherit the response unit. Dimensionless parameters have no unit.

Rule to document in user-facing docs:

> `bindcurve` does not perform automatic unit conversion. All concentration-like inputs must be provided in consistent units. Fitted concentration-like parameters are returned in the same units as the input concentrations.

## Default fitting strategy

The default fitting strategy should reflect biological dose-response practice.

For data with `n` independent experiments and `r` technical replicates, the default pipeline is:

```text
For each compound:
    For each independent experiment:
        aggregate technical replicates at each concentration
        fit one curve
    summarize fitted parameters across independent experiments
```

Thus, for `n = 3` and `r = 5`, the default behavior is three fitted curves per compound, each fitted to the aggregated technical replicates from one independent experiment.

This avoids pseudoreplication. Technical replicates mostly estimate point-level readout/pipetting noise, while independent experiments capture the variance that should dominate final biological uncertainty.

## Fitting strategies

The fitting layer should support at least three strategies.

### `per_experiment`

Recommended default.

For each compound and independent experiment, aggregate technical replicates at each concentration and fit one curve.

### `per_compound_summary`

Aggregate all observations for a compound by concentration, then fit one summary curve per compound.

This is useful for visualization and quick exploratory analysis, but it should not be the preferred source of final uncertainty.

### `pooled`

Fit all raw observations for a compound in one pooled fit.

This can be useful, but reported uncertainty from a single pooled fit can underestimate biological variability when independent experiments are treated as if all observations were independent.

## Replicate aggregation

Technical replicates should be aggregated before fitting in the default strategy.

Initial supported aggregation methods:

```text
mean
median
```

Technical replicate variance may be stored for diagnostics, but the final uncertainty for IC50/Kd-like values should usually come from variation between independent experiments.

## Parameter summarization

Concentration-like fitted parameters such as `IC50`, `EC50`, `Kd`, `Kds`, and `Kd3` should be summarized across independent experiments on the log scale when values are positive.

The default summary for concentration-like parameters should be based on `log10(parameter)` and back-transformed to produce a geometric mean and approximate confidence interval.

This is preferable to arithmetic mean +/- SD on the raw concentration scale because affinity and potency values are positive and often closer to log-normal than normal.

## Models

Models should be objects, not strings plus scattered `if model == ...` branches.

Each model should know:

- its name,
- its parameter specifications,
- which parameters are concentration-like,
- which parameters are response-like,
- how to evaluate `y = f(x, parameters)`,
- how to generate initial guesses,
- how to build lmfit parameters,
- how to compute residuals.

The base class is `BaseDoseResponseModel`.

The first implemented model is `IC50Model`.

## Model registry

A registry maps string names to model objects:

```python
model = get_model("ic50")
```

This allows a simple public API:

```python
results = bindcurve.fit(data, model="ic50")
```

while keeping the internal fitting architecture model-agnostic.

## Calculator

`FitCalculator` is the generic fitting coordinator.

It should not know model equations. It should only coordinate:

- data selection,
- replicate aggregation,
- experiment splitting,
- initial parameter generation,
- lmfit minimization,
- result construction,
- summary calculation,
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

A convenience wrapper should also exist:

```python
results = bc.fit(data, model="ic50", fixed={"ymin": 0.0, "ymax": 100.0})
```

## Settings

Large function signatures should be avoided. Fitting behavior should be controlled through settings objects.

Example:

```python
FitSettings(
    strategy="per_experiment",
    replicate_aggregation="mean",
    weighting="none",
    ci=False,
    ci_sigma=2.0,
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

or:

```python
fixed=None
```

## Results

Results should be structured and explicit.

Core result objects:

```text
ParameterEstimate
FitMetrics
FitResult
FitResults
```

`FitResult` represents one fitted curve. In the default strategy, that means one compound in one independent experiment.

`FitResults` is a collection of `FitResult` objects and provides dataframe exports for individual fits and compound-level summaries.

The result layer should preserve the raw lmfit result object for advanced users but expose a stable bindcurve-native interface for most use cases.

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

In collect mode, failed fits are returned as `FitResult` objects with:

```text
success = False
message = "..."
```

They should not silently disappear from the output.

## Plotting

Plotting should be separate from fitting.

Future plotting functions should consume:

```text
DoseResponseData + FitResults
```

and should not redo fitting logic internally.

Potential future functions:

```python
plot_curves(data, results)
plot_grid(data, results)
plot_residuals(results)
```

## Legacy compatibility

The old API may be kept temporarily through wrappers in a compatibility layer, but the new architecture should be the canonical API.

Legacy functions such as `fit_50`, `fit_Kd_direct`, and `fit_Kd_competition` should eventually become deprecated wrappers around the new implementation or be removed in a major release.

## First PR target

The first PR should implement only the foundation:

- `DoseResponseData`,
- `CompoundData`,
- `ParameterSpec`,
- `BaseDoseResponseModel`,
- `IC50Model`,
- `FitSettings`,
- `FitCalculator`,
- `ParameterEstimate`,
- `FitMetrics`,
- `FitResult`,
- `FitResults`,
- tests for synthetic IC50 fitting.

The first PR should not port every model. Additional models should be added after the skeleton is stable and tested.
