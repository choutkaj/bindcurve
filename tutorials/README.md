# bindcurve tutorials

These notebooks are compact, synthetic examples for the refactored bindcurve API.

Recommended order:

1. `00_basics.ipynb` — core `DoseResponseData -> fit -> FitResults -> plot` workflow
2. `01_ic50_models.ipynb` — IC50, EC50, and logIC50 logistic models
3. `02_direct_kd_models.ipynb` — direct-binding Kd models
4. `03_competitive_kd_models.ipynb` — competitive 3-state and 4-state Kd models, plus IC50-to-Kd conversion
5. `04_plotting.ipynb` — Axes-first plotting, confidence bands, asymptotes, curve points, and residuals

The examples generate synthetic data, so they do not require external files.

Run from the repository root after installing the package in editable mode:

```bash
python -m pip install -e .[test]
jupyter lab tutorials/
```
