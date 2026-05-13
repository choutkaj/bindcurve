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

## Notation

For one compound, BindCurve uses the following notation:

- `N_exp`: number of independent experiments.
- `N_conc`: number of assayed concentration points per experiment.
- `N_rep(e, c)`: number of technical replicates in experiment `e` at concentration `c`.

For balanced assay designs, this may be shortened to a constant `N_rep`.

Indices:

- `e = 1, ..., N_exp` for experiment.
- `c = 1, ..., N_conc` for concentration.
- `r = 1, ..., N_rep(e, c)` for replicate.

Response notation:

- `y_ecr`: raw response from experiment `e`, concentration `c`, replicate `r`.
- `ybar_ec`: arithmetic mean response within experiment `e` at concentration `c`.
- `ybar_c_grand`: grand mean response at concentration `c`, defined as the arithmetic mean of the experiment-level means contributing at that concentration.

## Numerical scale

`bindcurve` is unitless.

The numerical core accepts floats and does not perform automatic unit conversion. Users must provide all concentration-like values on a consistent numerical scale.

Fitted concentration-like parameters such as `IC50` and `Kd` are returned on that same numerical scale.

## Canonical fitting pipeline

`bindcurve` uses one canonical fitting pipeline.

For each compound:

1. Split data by independent experiment.
2. Aggregate technical replicates within each experiment at each concentration.
3. Fit one curve to each independent experiment.
4. Summarize fitted parameters across independent experiments.

For `N_exp = 3` and `N_rep = 5`, this yields three fitted curves per compound. This avoids pseudoreplication and treats inter-experiment variability as the main source of final biological uncertainty.

## Replicate aggregation

Technical replicates are aggregated before fitting.

Technical replicate responses are always aggregated using the arithmetic mean:

- within each experiment, `y_ecr` values are collapsed to `ybar_ec`;
- for `plot_compounds()`, the experiment-level means are then collapsed to `ybar_c_grand`.

Technical replicate variance may be stored for diagnostics, but final uncertainty for IC50/Kd-like values should usually come from variation between independent experiments.

For compound-level visualization in `plot_compounds()`, BindCurve also computes a plotting-only master fit from grand-mean response values. The grand mean is the arithmetic mean of experiment-level means at each concentration, so each independent experiment contributes equally regardless of replicate count.

## Uncertainty model

BindCurve distinguishes two uncertainty regimes:

- `intra-experiment uncertainty`: agreement or disagreement among technical replicates within one experiment at one concentration. This is the spread of `y_ecr` values around `ybar_ec` and is usually interpreted as measurement or instrument noise.
- `inter-experiment uncertainty`: agreement or disagreement among independent experiments. This is reflected by the spread of `ybar_ec` values across experiments at fixed concentration, and by the spread of fitted parameters such as `IC50` or `Kd` across experiment-level fits.

In BindCurve, inter-experiment uncertainty is treated as the primary biological uncertainty. Intra-experiment uncertainty is still useful for diagnostics and plotting, but it is not the main uncertainty used to summarize compound-level potency or affinity.

## Parameter summarization

Concentration-like fitted parameters such as `IC50`, `Kd`, `Kds`, and `Kd3` should be summarized across independent experiments on the log10 scale when all estimates are positive.

The package should report both simple linear summaries and geometric means for concentration-like parameters.

In short:

- response aggregation and plotting use arithmetic means;
- positive concentration-like parameter summaries across experiments use geometric means;
- non-concentration parameters such as `ymin`, `ymax`, and `hill_slope` use arithmetic means.

## Models

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
    settings=bc.FitSettings(),
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
    weighting="none",
    lmfit_method="leastsq",
    errors="raise",
)
```


## Results

Results are structured objects:

```text
ParameterEstimate
FitMetrics
FitResult
FitResults
```

`FitResult` represents one fitted curve: one compound in one independent experiment.

`FitResults` is a collection of `FitResult` objects and provides dataframe exports for individual fits and compound-level summaries. In compound-level summary tables, the count of independent experiment-level estimates is reported as `N_exp`.

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

Plotting should be separate from fitting. Most plotting functions should consume `DoseResponseData + FitResults` directly. `plot_compounds()` is an intentional exception: it may compute a plotting-only master fit from grand-mean response values in order to draw a single compound-level summary curve.

For the high-level wrapper plots:

- one plotted series should behave like one logical object, so markers and fitted curve share one legend label and one base color by default;
- `plot_fits()` is the experiment-level wrapper and keeps confidence bands as an option;
- `plot_compounds()` is the compound-summary wrapper and keeps asymptotes / arbitrary curve points out of scope;
- asymptotes and curve-point annotations should remain available through their dedicated plotting functions.

For uncertainty display:

- `plot_fits()` may show covariance-based pointwise confidence bands around experiment-level fitted mean curves.
- `plot_compounds()` should not show confidence bands. Compound-level uncertainty is communicated by SD/SEM error bars on grand-mean observations, reflecting inter-experiment variability.

Potential future functions:

```python
plot_curves(data, results)
plot_grid(data, results)
plot_residuals(results)
```
