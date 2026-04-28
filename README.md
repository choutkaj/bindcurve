# Welcome to BindCurve

[![PyPI Version](https://img.shields.io/pypi/v/bindcurve)](https://pypi.org/project/bindcurve/)
[![DOI](https://zenodo.org/badge/870812773.svg)](https://doi.org/10.5281/zenodo.15776819)
[![License: MIT](https://img.shields.io/github/license/choutkaj/bindcurve)](https://github.com/choutkaj/bindcurve/blob/main/LICENSE)

`bindcurve` is an lmfit-backed Python package for fitting dose-response and binding curves.

> [!NOTE]
> `bindcurve` is undergoing a major object-oriented refactor. Expect breaking API changes while the new architecture stabilizes.

## Current scope

The current refactored API focuses on dose-response data and one implemented model:

- `DoseResponseData`
- `IC50Model`
- generic `FitCalculator`
- structured `FitResults`

Additional dose-response, direct-binding, competitive-binding, and conversion models will be added after the core architecture is tested and stable.

## Installation

```bash
pip install bindcurve
```

For local development:

```bash
python -m pip install -e .[test]
pytest
```

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

## Architecture

See [`architecture.md`](architecture.md) for the intended design.

## How to cite

`bindcurve` has a DOI available from Zenodo. Please use the DOI or cite this repository directly.

## License

`bindcurve` is published under the MIT license.
