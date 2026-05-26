# Welcome to BindCurve

[![PyPI Version](https://img.shields.io/pypi/v/bindcurve)](https://pypi.org/project/bindcurve/)
[![Tests](https://github.com/choutkaj/bindcurve/actions/workflows/tests.yml/badge.svg)](https://github.com/choutkaj/bindcurve/actions/workflows/tests.yml)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://pypi.org/project/bindcurve/)
[![DOI](https://zenodo.org/badge/870812773.svg)](https://doi.org/10.5281/zenodo.15776819)
[![License: MIT](https://img.shields.io/github/license/choutkaj/bindcurve)](https://github.com/choutkaj/bindcurve/blob/main/LICENSE)

`bindcurve` is an lmfit-backed Python package for fitting dose-response curves.




## Current scope

The current API provides:

- dose-response data ingestion and validation through `DoseResponseData`
- a logistic `IC50Model`
- direct-binding models for simple, specific, and total Kd fitting
- competitive three-state and four-state binding models
- IC50-to-Kd conversion helpers
- structured fit results and plotting helpers for reports and figures

## Installation

```bash
uv add bindcurve
```

For local development:

```bash
uv python install 3.14
uv sync --group dev
uv run pytest
```

`bindcurve` supports Python 3.10, 3.11, 3.12, 3.13, and 3.14. Python 3.9 is not supported.

## Basic usage

`bindcurve` expects dose-response observations in long-form data:

```text
compound_id | experiment_id | concentration | replicate_id | response
```

Only these columns are required:

```text
compound_id | concentration | response
```

If `experiment_id` or `replicate_id` is missing, `bindcurve` fills defaults.

```python
import pandas as pd
import bindcurve as bc

raw = pd.DataFrame(
    {
        "compound_id": ["cmpd_a", "cmpd_a", "cmpd_a", "cmpd_a"],
        "experiment_id": ["exp1", "exp1", "exp1", "exp1"],
        "concentration": [0.01, 0.1, 1.0, 10.0],
        "response": [98.0, 85.0, 45.0, 5.0],
    }
)

data = bc.DoseResponseData.from_dataframe(
    raw,
)

results = bc.fit(
    data,
    model="ic50",
    fixed={"ymin": 0.0, "ymax": 100.0},
)

print(results.fits())
print(results.summary())
print(results.report(unit="uM"))
```

## Canonical fitting pipeline

`bindcurve` uses one canonical fitting pipeline.

For each compound, it:

1. splits observations by independent experiment;
2. averages technical replicates within each experiment at each concentration using the arithmetic mean;
3. fits one curve per independent experiment;
4. summarizes fitted parameters across independent experiments.

This avoids treating technical replicates as independent biological repeats.

At the high-level plotting layer, markers and fitted curves are coupled into one logical series by default: they share one legend label and one base color. `plot_fits()` can optionally draw covariance-based pointwise confidence bands around each experiment-level fitted mean curve. `plot_compounds()` draws grand-mean observations across experiments plus a separate plotting-only master fit, uses SD/SEM error bars instead of confidence bands to show inter-experiment uncertainty, and leaves asymptotes / curve points to their dedicated plotting helpers.

## Wide-format data

Wide assay tables can be normalized with `from_dataframe(..., format="wide")`:

```python
data = bc.DoseResponseData.from_dataframe(
    df,
    format="wide",
    compound_col="compound",
    concentration_col="dose",
    experiment_col="experiment",
    replicate_cols=["rep_1", "rep_2", "rep_3"],
)
```

`DoseResponseData` can also be serialized back to tabular or JSON representations:

```python
long_df = data.to_dataframe()
wide_df = data.to_dataframe(format="wide")

data.to_csv("dose_response_long.csv")
data.to_csv("dose_response_wide.csv", format="wide")

json_text = data.to_json(format="wide", indent=2)
reloaded = bc.DoseResponseData.from_json(json_text)
```

For a quick compound-level overview of the input dataset:

```python
summary = data.summary()
print(summary[["compound_id", "N_exp", "N_conc_total"]])
```

For simple compound-level data manipulation:

```python
subset = data.keep_only(["cmpd_a", 2])
trimmed = data.remove("cmpd_b")
merged = bc.DoseResponseData.concatenate(data_1, data_2)
```

## Direct model evaluation

You can also evaluate any registered model directly without fitting data:

```python
import numpy as np
import bindcurve as bc

model = bc.get_model("comp_3st_specific")
grid = np.logspace(-4, 1, 200)

evaluation = model.evaluate_components(
    grid,
    ymin=0.0,
    ymax=1.0,
    RT=0.05,
    LsT=0.005,
    Kds=0.02,
    Kd=1.6,
)

response = evaluation.response
free_receptor = evaluation.components["R_free"]
tracer_bound = evaluation.components["RLstar"]
```

`predict(...)` returns only the observable response, while
`evaluate_components(...)` returns a `ModelEvaluation` object with the response
plus any model-specific component arrays.

## Numerical scale

`bindcurve` is unitless.

The user is responsible for providing all concentration-like values on a consistent numerical scale. Fitted values such as `IC50` and `Kd` are returned on that same scale.

## Development

Common development commands:

```bash
uv sync --group dev
uv run pytest
uv run ruff check
uv build
```

The tutorial notebooks use the same `dev` environment, which includes `ipykernel`
and Jupyter support.

## Architecture

See [`architecture.md`](architecture.md) for the intended design.

## How to cite

`bindcurve` has a DOI available from Zenodo. Please use the DOI or cite this repository directly.

## License

`bindcurve` is published under the MIT license.
