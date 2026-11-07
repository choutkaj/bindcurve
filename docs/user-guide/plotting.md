---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Plotting

`bindcurve` has three plotting functions, all based on Matplotlib:

| Function | Shows |
| --- | --- |
| {py:func}`~bindcurve.plot_fits` | Each independent experiment: replicate means and fitted curve |
| {py:func}`~bindcurve.plot_compounds` | One summary curve per compound, across independent experiments |
| {py:func}`~bindcurve.plot_residuals` | Residuals of each fit against concentration |

The examples use the {download}`IC₅₀ tutorial dataset
<../tutorials/data/ic50.csv>`:

```{code-cell} python
import matplotlib.pyplot as plt

import bindcurve as bc

data = bc.DoseResponseData.from_csv("../tutorials/data/ic50.csv", format="wide")
results = bc.fit(data, "ic50")
```

## Axes and figures

Each function draws into a Matplotlib `Axes`, given with `ax=`, and returns it.
Without `ax`, a new figure is created. The functions set a logarithmic
concentration axis and label each curve, but leave axis labels, legends and
titles to you, so figures can be styled with the usual Matplotlib calls:

```python
fig, ax = plt.subplots(figsize=(5, 3.5))
bc.plot_fits(results, ax=ax)
ax.set_xlabel("inhibitor (nM)")
ax.set_ylabel("activity (% of control)")
ax.legend()
fig.savefig("fits.pdf")
```

## Individual experiments

`plot_fits` draws every successful fit with the replicate means of its data.
The error bars show the spread of the technical replicates at each
concentration, as SD (`errorbars="sd"`, the default), SEM (`"sem"`) or not at
all (`None`). They describe the measurement precision within one experiment,
not the variation between experiments.

`band=True` adds a pointwise confidence band around each curve, computed from
the fit's parameter covariance; `confidence` sets its level (default 0.95).
`compounds` and `experiments` restrict the plot to some fits:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3.5))
bc.plot_fits(results, compounds="inhibitor_a", band=True, ax=ax)
ax.set_xlabel("inhibitor (nM)")
ax.set_ylabel("activity (% of control)")
ax.legend()
plt.show()
```

## Compound summaries

`plot_compounds` draws one series per compound. Each marker is the mean of the
experiment means at that concentration, so every independent experiment counts
equally. The error bars show the spread between independent experiments, as
SD or SEM. The curve is the model evaluated at the summary parameters, so it
matches the values from `results.summary()` and `results.report()`:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3.5))
bc.plot_compounds(results, ax=ax)
ax.set_xlabel("inhibitor (nM)")
ax.set_ylabel("activity (% of control)")
ax.legend()
plt.show()
```

This is usually the figure to publish. State in the legend what the markers
and error bars show and how many independent experiments they come from, for
example "mean ± SD of three independent experiments, each in technical
triplicate".

## Residuals

`plot_residuals` plots the difference between each replicate mean and its
fitted curve. See [Checking fits](fit-quality.md#plots) for how to read it.

## Colors and several panels

`colors` takes one color, or one color per curve. Several plots can share a
figure by passing different axes:

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, compound in zip(axes, data.compounds):
    bc.plot_fits(results, compounds=compound, colors=["C0", "C1", "C2"], ax=ax)
    ax.set_title(compound)
    ax.set_xlabel("inhibitor (nM)")
axes[0].set_ylabel("activity (% of control)")
axes[0].legend()
plt.show()
```

## Showing individual technical replicates

To show the individual wells as well as the means, add them from the data
table with Matplotlib:

```python
table = data.table
wells = table[table["compound_id"] == "inhibitor_a"]
ax.scatter(wells["concentration"], wells["response"], s=8, color="0.6", zorder=0)
```
