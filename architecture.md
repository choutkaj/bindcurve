# BindCurve architecture

BindCurve fits dose-response and equilibrium-binding data through one canonical
pipeline. ITC, SPR, and kinetic traces are outside its scope.

## Package boundaries

- `datasets/dose_response.py` owns validated observations, selection, and the
  public import/export methods. Public tables and metadata are isolated copies.
- `datasets/formats.py` handles table layouts, column mappings, and JSON sources.
- `datasets/aggregation.py` computes arithmetic response means, sample SD, and SEM.
- `modeling/` owns equations, physical components, parameter specifications,
  initial guesses, and conversion between physical and optimizer coordinates.
- `fitting/` coordinates experiment-level fitting through lmfit. It contains no
  model-specific equations.
- `results/types.py` defines result records and uncertainty representations.
- `results/summaries.py` computes across-experiment parameter statistics.
- `results/core.py` validates result collections and exposes numerical tables.
- `results/reporting.py` formats manuscript-facing reports.
- `plotting/` renders observations, fitted predictions, residuals, confidence
  bands, and annotations. Plotting never performs a new fit.
- `conversion/` implements IC50-to-Kd conversions and their input validation.

## Scientific contract

The following rules are shared by fitting, summaries, and plotting. Architectural
cleanup must preserve them, along with model equations and numerical tolerances.

### Canonical observations

The canonical table is long-form:

```text
compound_id | experiment_id | concentration | replicate_id | response | metadata...
```

`compound_id`, `concentration`, and `response` are required. Missing experiment
identifiers default to `experiment_1`; missing replicate identifiers are generated
within each compound, experiment, and concentration group. Observations have unique
compound/experiment/concentration/replicate identities. Concentrations are finite
and positive; responses are finite. See [data formats](data_formats.md).

BindCurve is unitless. All concentration-like values supplied together must use
one consistent numerical scale. Fitted concentration parameters retain that scale.
Model evaluation also accepts zero concentration for evaluating limits.

### One fit per independent experiment

For each compound, the fitter:

1. Selects each independent experiment.
2. Averages technical replicate responses at each concentration arithmetically.
3. Generates initial guesses and applies fixed values and bounds.
4. Fits the experiment-level observations.
5. Summarizes successful fitted parameters across independent experiments.

Technical replicates do not count as independent experiments. A `FitResults`
collection rejects duplicate compound/experiment identities, inconsistent model
instances, incompatible parameter schemas, and inconsistent fixed parameters.

### Known observation uncertainty

Input may contain either `sigma` (known observation standard deviation) or
`weight` (reciprocal standard deviation). Both must be finite and positive.
For independent replicate errors, uncertainty of an arithmetic mean is propagated
as `sqrt(sum(sigma_i**2)) / n`. Fitting standardizes residuals by this propagated
sigma. Without known sigma, fitting uses unweighted residuals. Empirical replicate
SD or SEM is not substituted for known observation sigma.

Fit diagnostics distinguish RSS and reduced RSS from chi-square and reduced
chi-square. Chi-square is available only when observation uncertainty is known.
The known-sigma likelihood includes its Gaussian normalization. Optimizer
covariance is transformed back to public physical parameter coordinates.

### Parameter summaries

Native additive parameters use arithmetic means, sample SD (`ddof=1`),
`SEM = SD / sqrt(N_exp)`, and two-sided Student-t 95% confidence intervals.

Positive concentration parameters use those same statistics on `log10` values.
Their linear center is `10**log10_mean`; linear SD, SEM, and CI95 intervals are
back-transformed log intervals. Linear concentration uncertainty is consequently
asymmetric. With one successful experiment, spread and confidence intervals are
unavailable. Fixed parameters are excluded from estimated-parameter summaries.

`parameter_values()` returns arithmetic means for varying native parameters,
geometric centers for varying concentration parameters, and the common values
of fixed parameters. It does not fit another curve.

### Plotting

`plot_fits()` displays observations and predictions for successful experiment-level
fits. Optional bands are covariance-based pointwise confidence bands around each
fitted mean curve, using a Student-t multiplier.

`plot_compounds()` draws the pointwise arithmetic mean of successful experiment
predictions. Its grand-mean observations are arithmetic means of experiment means,
so experiments contribute equally regardless of their technical replicate counts.
Its SD/SEM error bars describe inter-experiment response variability. Failed fits
are excluded from the prediction average; observations retain the selected data.
Compound plots do not have confidence bands.

Each plotted series shares one base color and legend entry across markers and
curves. Asymptotes and arbitrary curve points have dedicated annotation functions.

## Public fitting and results API

```python
import bindcurve as bc

results = bc.fit(
    data,
    model="ic50",
    settings=bc.FitSettings(errors="raise"),
    fixed={"ymin": 0.0, "ymax": 100.0},
)
```

`get_model(name)` resolves built-in models. `fit()` also accepts a custom
`BaseDoseResponseModel` instance, which is retained in its results. The calculator
is an internal implementation detail.

- `fit_summary()` provides per-experiment estimates, numerical metrics, optimizer
  messages, and failure details.
- `fixed_parameters()` lists fixed values separately.
- `parameters()` provides long-form native and concentration summaries, including
  canonical log10 concentration statistics.
- `summary()` provides one row per compound, experiment and fit counts,
  observation counts, parameter centers/intervals, RSS, and chi-square.
- `report()` formats a selected concentration summary in linear, log, or both
  representations. Invalid options are rejected even if all fits failed.

The default error mode re-raises fitting exceptions. `FitSettings(errors="collect")`
records exceptions as failed `FitResult` objects and continues with other
experiments. Optimizer-reported failures retain their diagnostic context. Failed
fits remain visible in result tables and are excluded from parameter summaries.

## Maintenance

The maintained user documentation lives in `docs/` and is built with Sphinx.
`quarto_website/` and `docs_legacy/` are historical material, not current API
references. The root architecture and data-format documents describe the current
implementation.

Preserve independent equilibrium, mass-balance, concentration-scale, cancellation,
weighting, and uncertainty tests when refactoring. Numerical root selection,
stability expressions, and solver tolerances are scientific implementation
choices, not formatting opportunities.
