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

`bindcurve` accepts long-form observations. The required columns are
`compound_id`, `concentration`, and `response`:

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

data = bc.DoseResponseData.from_dataframe(observations)
results = bc.fit(
    data,
    model="ic50",
    fixed={"ymin": 0.0, "ymax": 100.0},
)

print(results.summary()[["compound_id", "IC50"]])
```

```text
  compound_id  IC50
0     example   1.0
```

Concentrations must use one consistent unit. The fitted IC₅₀ is reported in
that same unit. See the [logistic-model theory](theory/logistic.md) for the
model definition and interpretation.
