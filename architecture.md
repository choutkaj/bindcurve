# bindcurve architecture

bindcurve fits dose-response and equilibrium-binding curves to plate-style
titration data. ITC, SPR and kinetic traces are out of scope.

## Modules

| Module | Responsibility |
| --- | --- |
| `data.py` | `DoseResponseData`: validated long-form observations, the wide layout, and `replicate_means`. |
| `models/` | `Model`, `BindingModel` and `Parameter`; one module per equation family; the `MODELS` registry. |
| `fitting.py` | `fit()`: per-experiment least squares, covariance, quality warnings. |
| `results.py` | `FitResult`, `FitResults`, across-experiment statistics and reports. |
| `plotting.py` | `plot_fits`, `plot_compounds`, `plot_residuals`. Plotting never fits. |
| `conversion.py` | Vectorized IC50-to-Kd conversions. |

Dependencies point one way: `data` and `models` are independent;
`results` uses both; `fitting` and `plotting` use `results`.

## Scientific contract

Refactoring must preserve the following, together with the model equations,
their solvers and numerical tolerances.

### Models

Every model maps a fraction onto two plateaus,
`ymin + (ymax - ymin) * fraction(x)`, on the untransformed concentration axis.
Binding models expose their equilibrium species through `species()`. Root
selection and cancellation-free forms in `models/` are scientific choices:
the direct quadratic uses the rationalized root, and the competitive models
bracket the receptor balance normalized by `RT`, which has exactly one root.
`Parameter(concentration=True)` marks positive concentrations; `fixed=True`
marks assay constants that `fit()` requires.

### Fitting

`fit()` fits each compound/experiment separately:

1. Technical replicates are averaged per concentration.
2. Each mean is weighted by its replicate count (equivalent to fitting the
   replicates) and the covariance is scaled by the residual scatter.
3. Concentrations are optimized as log10 values within 1e-100 to 1e100 by
   `scipy.optimize.least_squares`; residuals and plateaus are measured in
   units of the response range, so results do not depend on the response
   scale or baseline. Covariance is `inv(J.T J)` via SVD, `None`
   if `J` is rank deficient, and is transformed back to linear values.
4. Converged fits are flagged when standard errors are missing, a fitted
   concentration lies outside the tested range, or its standard error
   exceeds it. `fit()` emits one `UserWarning` for flagged fits; they stay in
   summaries.

Errors re-raise by default; `errors="collect"` records failed fits.

### Summaries

Across successful experiments, native parameters get the mean, sample SD,
SEM and Student-t 95% CI. Concentration parameters get the same on log10
values; their center is the geometric mean and the CI is back-transformed.
`parameters()` returns these centers plus the fixed values, and
`plot_compounds()` draws the model there, so plots match `report()`.
Confidence bands are pointwise delta-method bands using the Student-t
quantile.

## Testing

`tests/references.py` holds references that never call bindcurve's solvers:
high-precision bisection, simultaneous mass-balance solutions, data built
from free species, and exact IC50s. Keep expectations analytic or
independently computed; avoid optimizer-output snapshots and pixel tests.

Run the suite with `uv run --group test pytest`.
