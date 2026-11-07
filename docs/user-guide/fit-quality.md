---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Checking fits

Check every fit before using its parameters. A fit can converge and still
describe the data poorly, or return a parameter the data do not determine.

The examples on this page use the {download}`IC₅₀ tutorial dataset
<../tutorials/data/ic50.csv>`: two inhibitors, three independent experiments
each, with technical triplicates.

```{code-cell} python
import matplotlib.pyplot as plt

import bindcurve as bc

data = bc.DoseResponseData.from_csv("../tutorials/data/ic50.csv", format="wide")
results = bc.fit(data, "ic50")
```

## The experiment table

`results.experiments()` has one row per fit, that is, per independent
experiment of each compound:

```{code-cell} python
results.experiments()
```

| Column | Meaning |
| --- | --- |
| `success` | Whether the optimizer converged |
| `message` | Optimizer or error message |
| `warnings` | Quality warnings for converged fits, described below |
| `n_data` | Number of concentrations fitted |
| `rss` | Residual sum of squares of the replicate means |
| `r_squared` | Coefficient of determination of the replicate means |
| *parameter* | Fitted value of each free parameter |
| *parameter*`_stderr` | Its standard error within this experiment |

$R^2$ is close to 1 for almost any sigmoidal data set and says little about
whether the model is appropriate. Use it together with the plots below, not on
its own.

## Quality warnings

`bindcurve` flags converged fits whose estimates may not be supported by the
data. `fit` then emits a `UserWarning` with the number of flagged fits, the
`warnings` column gives the reasons, and the `N_flagged` column of
`results.summary()` counts them per compound.

| Warning | Meaning | What to do |
| --- | --- | --- |
| *name* lies outside the tested concentration range. | The IC₅₀ or $K_d$ was extrapolated beyond the highest or lowest concentration. | Extend the concentration range. Otherwise report the value only as above or below the tested range. |
| *name* is poorly determined: its standard error exceeds the estimate. | The data constrain the parameter only weakly, typically because a plateau or the midpoint is not well covered. | Add concentrations around the midpoint and on both plateaus, or fix a parameter whose value is independently known. |
| Standard errors could not be estimated. | Some parameters cannot be separated from each other with these data, for example flat data or too few concentrations. | Check the data and the concentration range, and consider fixing a parameter whose value is known. |

Flagged fits are still included in the summary statistics. To leave one out,
remove its rows from the data and fit again:

```python
table = data.table
keep = ~((table["compound_id"] == "inhibitor_b") & (table["experiment_id"] == "exp_2"))
results = bc.fit(bc.DoseResponseData(table[keep]), "ic50")
```

## Plots

Plot each fit over its data. The replicate means should lie on the curve, with
both plateaus and the midpoint covered by measured concentrations:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3.5))
bc.plot_fits(results, compounds="inhibitor_a", ax=ax)
ax.set_xlabel("inhibitor (nM)")
ax.set_ylabel("activity (% of control)")
ax.legend()
plt.show()
```

{term}`Residuals <Residual>` show systematic deviations more clearly. For a
good fit they scatter randomly around zero. A trend, such as residuals that
are positive at both ends and negative in the middle, means that the model
does not describe the shape of the curve:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 2.5))
bc.plot_residuals(results, compounds="inhibitor_a", ax=ax)
ax.set_xlabel("inhibitor (nM)")
ax.set_ylabel("residual (%)")
ax.legend()
plt.show()
```

## Common problems

- **No lower or upper plateau.** The concentration range does not cover the
  whole curve. The IC₅₀ is then poorly determined; extend the range, or fix the
  missing plateau if its value is known independently.
- **Hill slope far from 1.** Very steep or shallow curves can come from
  compound aggregation, solubility limits, depletion or more complex binding.
  Check the data before interpreting the IC₅₀.
- **Systematic residuals.** The model does not match the data, for example a
  three-state model applied to incomplete displacement. Consider a different
  [model](models.md).
- **Large differences between independent experiments.** Compare the
  experiment-level values in `results.experiments()`. An outlying experiment
  can have a technical cause, such as a pipetting or dilution error.
