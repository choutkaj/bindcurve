# bindcurve tutorials

These notebooks are compact, synthetic examples for the refactored bindcurve API.

Recommended order:

1. `00_basics.ipynb` - core `DoseResponseData -> fit -> FitResults -> plot` workflow
2. `07_data_handling.ipynb` - subsetting, removing, concatenating, and round-tripping `DoseResponseData`
3. `01_ic50_models.ipynb` - IC50 logistic model
4. `02_direct_kd_models.ipynb` - direct-binding Kd models
5. `03_competitive_kd_models.ipynb` - competitive 3-state and 4-state Kd models, plus IC50-to-Kd conversion
6. `04_plotting.ipynb` - Axes-first plotting, confidence bands, asymptotes, curve points, and residuals
7. `05_plotting_components.ipynb` - parameter playground for hidden component curves in `dir_specific` and `comp_3st_specific`
8. `06_galectin.ipynb` - real galectin binding data workflow

The notebooks either use synthetic data or pure parameter playgrounds, so they do not require external files.

Run from the repository root after syncing the development environment:

```bash
uv sync --group dev
uv run jupyter lab tutorials/
```
