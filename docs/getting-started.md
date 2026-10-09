# Getting started

`bindcurve` supports Python 3.10 through 3.14. Install it into your project
environment with either `uv` or `pip`:

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

## Fit an inhibition curve

```python
import bindcurve as bc
import pandas as pd

observations = pd.DataFrame(
    {
        "compound_id": ["example"] * 7,
        "concentration": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
        "response": [99.0, 90.9, 66.7, 50.0, 33.3, 9.1, 1.0],
    }
)

data = bc.DoseResponseData(observations)
results = bc.fit(data, "ic50", fixed={"ymin": 0.0, "ymax": 100.0})

print(results.summary()[["compound_id", "IC50"]])
```

```text
  compound_id  IC50
0     example   1.0
```

`fit` fits every independent experiment of every compound separately.
`results.experiments()` lists the experiment-level fits, `results.summary()`
summarizes them per compound, `results.report()` formats the potency for a
manuscript, and `bc.plot_fits(results)` draws them. See the
[logistic-model theory](theory/logistic.md) for the model itself.

## Input data

Observations are long-form, one row per measured response:

```text
compound_id,experiment_id,concentration,response
Cmpd_1,exp_1,0.001,98.1
Cmpd_1,exp_1,0.001,97.5
Cmpd_1,exp_1,0.003,94.2
Cmpd_1,exp_2,0.001,97.9
```

- `compound_id`, `concentration` and `response` are required.
- `experiment_id` identifies independent experiments; it defaults to a single
  experiment. Rows sharing compound, experiment and concentration are
  technical replicates, which are averaged before fitting.
- `sigma`, if present, is the known absolute standard deviation of each
  response. It is not the empirical replicate SD or SEM.
- Concentrations must be positive and share one unit; fitted concentrations
  are reported in that unit. Other columns are kept but not used.

The wide layout has one row per concentration and replicate responses in
columns starting with `response_`:

```text
compound_id,experiment_id,concentration,response_1,response_2,response_3
Cmpd_1,exp_1,0.001,98.1,97.5,99.0
Cmpd_1,exp_1,0.003,94.2,95.0,93.7
```

```python
data = bc.DoseResponseData.from_csv("observations.csv")
data = bc.DoseResponseData.from_csv("observations-wide.csv", format="wide")
```

Rename columns with pandas if your files use other names, and use
`data.select(...)` to fit a subset of compounds.
