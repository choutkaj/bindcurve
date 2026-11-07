---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

`bindcurve` fits dose-response and binding curves. It estimates
{term}`IC₅₀` values with a four-parameter logistic model, and
{term}`dissociation constants <Dissociation constant>` from direct and
competitive binding experiments. Each {term}`independent experiment` is
fitted separately, and the results are then summarized per compound.

## Installation

`bindcurve` needs Python 3.10 or newer. Install it with `uv` or `pip`:

::::{tab-set}
:::{tab-item} uv

```console
uv add bindcurve
```

:::
:::{tab-item} pip

```console
python -m pip install bindcurve
```

:::
::::

The examples in this documentation use `matplotlib` for figures, which is
installed with `bindcurve`. They run in a Python script or in a Jupyter
notebook.

## A first fit

The data below come from a single inhibition experiment: enzyme activity, in
percent of the uninhibited control, measured at nine inhibitor concentrations
in nM. `bindcurve` expects a table with one row per measured response:

```{code-cell} python
import matplotlib.pyplot as plt
import pandas as pd

import bindcurve as bc

observations = pd.DataFrame(
    {
        "compound_id": "inhibitor",
        "concentration": [1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        "response": [99.6, 99.4, 95.7, 87.7, 71.1, 44.6, 21.7, 12.2, 3.4],
    }
)
data = bc.DoseResponseData(observations)
```

`DoseResponseData` checks the table: required columns, numeric responses, and
positive concentrations. Fit the logistic model, named `"ic50"`, and look at
the fitted parameters:

```{code-cell} python
results = bc.fit(data, "ic50")
results.experiments()[["compound_id", "IC50", "IC50_stderr", "hill_slope", "r_squared"]]
```

The IC₅₀ is about 235 nM, in the same unit as the input concentrations, with a
standard error of about 14 nM. The {term}`Hill slope` is close to 1.

Plot the data and the fitted curve:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3.5))
bc.plot_fits(results, ax=ax)
ax.set_xlabel("inhibitor (nM)")
ax.set_ylabel("activity (% of control)")
plt.show()
```

## How the data are processed

Real data usually have several {term}`technical replicates <Technical replicate>`
(for example triplicate wells) within each
{term}`independent experiment`, and several independent experiments per
compound. `bindcurve` handles them in three steps:

1. Technical replicates, rows with the same compound, experiment and
   concentration, are averaged.
2. The model is fitted to the replicate means of each independent experiment
   separately. `results.experiments()` lists these fits.
3. `results.summary()` combines the experiment-level estimates of each
   compound into a mean, standard deviation and 95% confidence interval, with
   the independent experiments as $n$.

The example above is a single experiment, so its summary repeats that fit and
has no spread.

## Next steps

- [Input data](user-guide/data.md) describes the table format, CSV files,
  technical replicates and independent experiments in detail.
- The [tutorials](tutorials/index.md) work through complete IC₅₀,
  direct-binding and competitive-binding analyses.
- The [theory](theory/index.md) section defines every model.
