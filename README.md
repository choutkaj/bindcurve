# Welcome to BindCurve

[![PyPI Version](https://img.shields.io/pypi/v/bindcurve)](https://pypi.org/project/bindcurve/)
[![Tests](https://github.com/choutkaj/bindcurve/actions/workflows/tests.yml/badge.svg)](https://github.com/choutkaj/bindcurve/actions/workflows/tests.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://pypi.org/project/bindcurve/)
[![DOI](https://zenodo.org/badge/870812773.svg)](https://doi.org/10.5281/zenodo.15776819)
[![License: MIT](https://img.shields.io/github/license/choutkaj/bindcurve)](https://github.com/choutkaj/bindcurve/blob/main/LICENSE)

`bindcurve` is an lmfit-backed Python package for fitting dose-response curves.

<p align="center">
  <img src="assets/logo/logo1.png" alt="bindcurve logo" width="400">
</p>


## Current scope

The current API provides:

- dose-response data ingestion and validation through `DoseResponseData`
- logistic models such as `IC50Model`, `LogIC50Model`, and `EC50Model`
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
uv python install 3.12
uv sync --group dev
uv run pytest
```

`bindcurve` supports Python 3.10, 3.11, and 3.12. Python 3.9 is not supported.

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
    concentration_unit="uM",
    response_unit="percent",
)

results = bc.fit(
    data,
    model="ic50",
    fixed={"ymin": 0.0, "ymax": 100.0},
)

print(results.fits_to_dataframe())
print(results.summary_to_dataframe())
```

## Default fitting strategy

The default strategy is `per_experiment`.

For each compound, `bindcurve`:

1. splits observations by independent experiment;
2. aggregates technical replicates at each concentration;
3. fits one curve per independent experiment;
4. summarizes fitted parameters across independent experiments.

This avoids treating technical replicates as independent biological repeats.

Alternative strategies are available through `FitSettings`:

```python
bc.FitSettings(strategy="per_experiment")
bc.FitSettings(strategy="pooled")
bc.FitSettings(strategy="per_compound_summary")
```

## Wide-format data

Wide assay tables can be normalized with `from_wide_dataframe`:

```python
data = bc.DoseResponseData.from_wide_dataframe(
    df,
    compound_col="compound",
    concentration_col="dose",
    experiment_col="experiment",
    replicate_cols=["rep_1", "rep_2", "rep_3"],
)
```

## Units

`bindcurve` is unitless by computation and unit-aware by annotation.

The user is responsible for providing all concentration-like values in consistent units. Fitted concentration-like parameters are reported in the same unit label as the input concentrations.

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
