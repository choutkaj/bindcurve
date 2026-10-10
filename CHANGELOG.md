# Changelog

## 0.3.0 (2026-10-10)

A rewrite into a small, flat package with a lean API. It is not backward
compatible with 0.2.0; see the migration notes below.

### Changed

- Fitting uses `scipy.optimize.least_squares` directly; `lmfit` is no longer a
  dependency. Concentration parameters are optimized as log10 values.
- `fit(data, model, *, fixed=None, bounds=None, errors="raise")` fits every
  experiment of every compound separately. Technical replicates are averaged
  per concentration, each mean is weighted by its replicate count, and
  standard errors are scaled by the residual scatter.
- Converged fits that the data may not support are flagged in
  `FitResult.warnings`, and `fit()` emits one `UserWarning` for them. They are
  kept in summaries and counted as `N_flagged`.
- IC50-to-Kd conversions accept scalars or arrays, e.g. a column of
  `FitResults.summary()`, and return NaN for IC50 values that are physically
  incompatible with the assay constants.
- Minimum dependency versions: numpy 1.23.5, pandas 1.5.3, scipy 1.9.3 and
  matplotlib 3.6.3. The Python upper bound (`<3.15`) is dropped.

### Fixed

- Fit results no longer depend on the magnitude or baseline of the response.
  Small responses (around 1e-4 and below) or a baseline much larger than the
  response window previously stopped the optimizer early without a warning.

### Removed

- Known observation uncertainties: `sigma` and `weight` columns are no longer
  used, along with `chi_square` and standardized residuals.
- Data and result quality reports and dashboards, `DataQualityThresholds` and
  `ResultQualityThresholds`.
- `plot_asymptotes`, `plot_curve_points` and `CurvePoint`.
- JSON input and output, and CSV export of `DoseResponseData`.

### Migrating from 0.2.0

Model names (`"ic50"`, `"dir_simple"`, `"comp_4st_total"`, ...) are unchanged.

| 0.2.0 | 0.3.0 |
| --- | --- |
| `fit(data, model="ic50", settings=FitSettings(errors="collect"))` | `fit(data, "ic50", errors="collect")` |
| `fit(data, compounds=[...])` | `fit(data.select([...]), ...)` |
| `DoseResponseData.from_dataframe(df)` | `DoseResponseData(df)` |
| `DoseResponseData.from_dataframe(df, format="wide")` | `DoseResponseData.from_wide(df)` |
| `from_csv(..., replicate_prefix=...)` | `from_csv(..., format="wide", prefix=...)` |
| `compound_col=`, `response_col=`, ... | Rename columns with `DataFrame.rename` first |
| `keep_only(...)` | `select(...)` |
| `data.to_dataframe()` | `data.table` |
| `cheng_prusoff_corrected(...)` | `munson_rodbard(...)` |
| `convert_ic50_to_kd(...)`, `IC50ConversionResult` | Call `cheng_prusoff`, `munson_rodbard` or `coleska` |
| `BaseDoseResponseModel`, `ParameterSpec` | `Model` / `BindingModel`, `Parameter` |
| `ModelEvaluation` | `BindingModel.species()` |
| `results.parameter_values(compound_id)` | `results.parameters(compound_id)` |
| `results.fit_summary()` | `results.experiments()` |
| `results.fit_results` | `results.fits` |
| `results.successful()`, `results.failed()` | Filter `results.fits` by `fit.success` |
