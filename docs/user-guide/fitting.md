# Fitting

{py:func}`bindcurve.fit` fits one model to every independent experiment of
every compound in a data object:

```python
import bindcurve as bc

results = bc.fit(data, "ic50")
```

It returns a {py:class}`~bindcurve.FitResults` object, described in
[Checking fits](fit-quality.md) and [Results and reporting](results.md).

## What a fit does

For each {term}`independent experiment` of each compound, `fit`:

1. averages the {term}`technical replicates <Technical replicate>` at each
   concentration;
2. fits the model to these replicate means by nonlinear least squares, giving
   each mean a weight proportional to its number of technical replicates;
3. estimates a standard error for each fitted parameter.

The fit needs more concentrations than fitted parameters. The `ic50` model
with all four parameters free therefore needs at least five concentrations,
although considerably more are needed in practice to define both plateaus and
the midpoint.

Starting values are chosen automatically: the plateaus from the lowest and
highest replicate means, concentration parameters such as `IC50` or `Kd` from
the concentration whose response is closest to halfway, and a Hill slope of 1.
Concentration parameters are fitted on a $\log_{10}$ scale, which keeps them
positive.

### Standard errors and variation between experiments

The standard error of a single fit is estimated from the scatter of the
replicate means around the fitted curve. It describes how precisely that one
experiment determines the parameter. It does not include the variation
between independent experiments, which is usually larger. That variation
appears only when several independent experiments are combined in
`results.summary()`, which is why the summary, not a single fit, should be
reported.

## Fixed values

Pass fixed parameter values as a dictionary:

```python
results = bc.fit(
    data,
    "comp_3st_specific",
    fixed={"RT": 40.0, "LsT": 10.0, "Kds": 15.0},
)
```

The binding models require their {term}`assay constants <Assay constant>`
here; `fit` raises an error if one is missing. All concentrations, fixed or
fitted, must use the same unit.

### Fixing plateaus or the Hill slope

The plateaus `ymin` and `ymax` and the Hill slope are fitted by default, and
they should normally stay that way. Fixing them is justified when the data
cannot determine them and their value is known independently, for example:

- A competition curve does not reach its lower plateau within the tested
  concentrations. In the three-state models, `ymin` is the signal of free
  tracer, which can be taken from a direct-binding fit with the same tracer
  and instrument settings.
- Responses are normalized to controls that define the plateaus, and the curve
  is too incomplete to estimate one of them.

```python
results = bc.fit(data, "ic50", fixed={"ymin": 0.0})
```

Fixing a plateau or slope to a value that does not match the data biases the
other parameters, including the IC₅₀ or $K_d$, and makes their standard errors
too small. Do not fix plateaus at 0 and 100% only because the data were
normalized.

## Bounds

Fitted parameters can be limited to a range with `bounds`, given as
`{name: (lower, upper)}`. Use `None` for a side without a limit:

```python
results = bc.fit(data, "ic50", bounds={"hill_slope": (0.5, 3.0)})
```

Bounds for concentration parameters must be positive. A parameter cannot be
both fixed and bounded. A fit that ends on a bound usually means that the data
do not determine that parameter.

## Failed fits

By default, an error in any one fit stops `fit`. With `errors="collect"`,
failed fits are recorded and the remaining experiments are still fitted:

```python
results = bc.fit(data, "ic50", errors="collect")
```

Failed fits have `success` set to `False` and an error `message` in
`results.experiments()`. They are left out of the summary statistics and
counted in its `N_failed` column. Converged fits that may not be supported by
the data are flagged with warnings, as described in
[Checking fits](fit-quality.md).
