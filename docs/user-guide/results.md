---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Results and reporting

`fit` returns a {py:class}`~bindcurve.FitResults` object. Its `experiments()`
table, with one row per independent experiment, is described in
[Checking fits](fit-quality.md). This page covers the per-compound summary,
formatted reports and IC₅₀-to-$K_d$ conversion.

The examples use the {download}`IC₅₀ tutorial dataset
<../tutorials/data/ic50.csv>`:

```{code-cell} python
import bindcurve as bc

data = bc.DoseResponseData.from_csv("../tutorials/data/ic50.csv", format="wide")
results = bc.fit(data, "ic50")
```

## Summary across independent experiments

`results.summary()` combines the successful fits of each compound. The
independent experiments are the $n$ of these statistics; technical replicates
were already averaged within each experiment.

```{code-cell} python
summary = results.summary()
summary[["compound_id", "N_fit", "IC50", "logIC50", "logIC50_SD",
         "IC50_CI95_lower", "IC50_CI95_upper"]]
```

The count columns are:

| Column | Meaning |
| --- | --- |
| `N_fit` | Number of independent experiments with a successful fit, the $n$ |
| `N_failed` | Number of failed fits, excluded from the statistics |
| `N_flagged` | Number of successful fits with quality warnings, included in the statistics |

For each fitted parameter there are a central value and its spread.
{term}`Concentration-like parameters <Concentration-like parameter>`, such as
`IC50` or `Kd`, are summarized on the $\log_{10}$ scale, because their values
from independent experiments are approximately log-normally distributed:

| Column | Meaning |
| --- | --- |
| `IC50` | {term}`Geometric mean <Geometric mean>`, $10^{\mathtt{logIC50}}$ |
| `logIC50` | Mean of $\log_{10}(\mathrm{IC}_{50})$, in the input unit |
| `logIC50_SD` | Standard deviation of $\log_{10}(\mathrm{IC}_{50})$ |
| `logIC50_SEM` | Standard error of the mean of $\log_{10}(\mathrm{IC}_{50})$ |
| `IC50_CI95_lower`, `IC50_CI95_upper` | 95% confidence interval of the geometric mean |

Other parameters, such as `ymin`, `ymax` and `hill_slope`, get the arithmetic
mean and the columns `_SD`, `_SEM`, `_CI95_lower` and `_CI95_upper`.

The three measures of spread answer different questions:

- The **SD** describes how much the values of independent experiments differ,
  that is, the reproducibility of the assay.
- The **SEM** is $\mathrm{SD}/\sqrt{n}$ and describes how precisely the mean is
  known.
- The **95% confidence interval** is computed on the $\log_{10}$ scale with
  Student's $t$ distribution and $n-1$ degrees of freedom, then transformed
  back. It is therefore asymmetric around the geometric mean. With few
  experiments it is wide: for $n=3$ its half-width is about 2.5 SD on the
  log scale.

The SD, SEM and confidence interval need at least two successful fits and are
missing otherwise.

## Formatted reports

`results.report()` formats one concentration parameter per compound as text:

```{code-cell} python
results.report(unit="nM")
```

By default it reports the geometric mean, with the range of one SD on the log
scale, $[10^{m-s}, 10^{m+s}]$, to two significant figures. The options are:

| Argument | Values | Effect |
| --- | --- | --- |
| `parameter` | parameter name | Parameter to report; required when a model fits several concentration parameters |
| `uncertainty` | `"sd"`, `"sem"`, `"ci95"` | Spread shown with the central value |
| `log` | `True`, `False` | Report $\log_{10}$ values, as mean ± spread |
| `digits` | integer | Significant figures for linear values, decimals for log values |
| `unit` | text | Unit appended to linear values |

```{code-cell} python
results.report(uncertainty="ci95", unit="nM")
```

### Reporting conventions

The reporting guidelines listed under [References](#references) recommend:

- State $n$ and the type of replicate, for example "three independent
  experiments, each in technical triplicate". Do not derive statistics from
  technical replicates.
- Report potencies to two significant figures. A well-run assay reproduces
  potency to within a factor of 2–3, so more digits are not meaningful.
- Report the mean and SD or SEM on the log scale, as pIC₅₀ with two decimals
  and the SD or SEM with one decimal. For linear IC₅₀ or $K_d$ values, report
  the 95% confidence interval.
- Define every error bar in figure legends.

`report(log=True)` gives $\log_{10}$ values in the input unit. They equal
$-\mathrm{pIC}_{50}$ only if the concentrations are in mol/L; for nM data,
$\mathrm{pIC}_{50} = 9 - \log_{10}(\mathrm{IC}_{50}/\mathrm{nM})$. It prints
the spread with the same number of decimals as the mean, so round the SD or
SEM to one decimal when writing it up.

An example of a matching methods sentence:

> IC₅₀ values are geometric means with 95% confidence intervals from three
> independent experiments, each measured in technical triplicate. Curves were
> fitted with a four-parameter logistic model.

## Using the results

`summary()` and `experiments()` return pandas DataFrames, which can be saved
with `to_csv()` or `to_excel()`. `results.parameters(compound_id)` returns the
summary values of one compound together with the fixed values, ready for
`results.model.evaluate`:

```{code-cell} python
results.parameters("inhibitor_a")
```

(converting-an-ic50-to-kd)=
## Converting an IC₅₀ to Kd

For one-site competitive binding at equilibrium, an IC₅₀ can be converted to
the competitor's $K_d$ with one of three functions. They accept single values
or arrays, so a whole summary column can be converted at once:

| Function | Needs | Notes |
| --- | --- | --- |
| {py:func}`~bindcurve.cheng_prusoff` | `LsT`, `Kds` | Approximation for low receptor concentration |
| {py:func}`~bindcurve.munson_rodbard` | `LsT`, `Kds`, `y0` | Exact; `y0` is the bound-to-free tracer ratio without competitor |
| {py:func}`~bindcurve.coleska` | `RT`, `LsT`, `Kds` | Exact; computes the competitor-free state from the assay constants |

```python
kd = bc.coleska(summary["IC50"], RT=40.0, LsT=10.0, Kds=15.0)
```

Inputs that are inconsistent with the binding model give NaN. The conversion
applies to the value it is given, so converting a summary IC₅₀ gives a $K_d$
without a spread. When the assay constants are known, fitting a
[competitive-binding model](models.md) instead gives $K_d$ for each
independent experiment and the full summary statistics. The
[conversion theory](../theory/conversions.md) describes when each conversion
applies.

## References

- Nature Communications, [Guidance on technical vs biological
  replicates](https://www.nature.com/documents/Biological_and_technical_replicates_guidelines.pdf).
- Heppner, D. E. et al. (2026), [Consistent quantitative reporting in medicinal
  chemistry](https://doi.org/10.1021/acs.jmedchem.6c01807). *J. Med. Chem.*
  69, 16115–16117.
