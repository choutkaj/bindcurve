# Input data

All fitting starts from a {py:class}`~bindcurve.DoseResponseData` object, which
holds the measured responses of one or more compounds in a validated table.

## Table format

The table is in long format, with one row per measured response:

```text
compound_id,experiment_id,concentration,response
inhibitor_a,exp_1,1.0,99.1
inhibitor_a,exp_1,1.0,98.4
inhibitor_a,exp_1,3.0,97.0
...
inhibitor_a,exp_2,1.0,100.3
```

| Column | Required | Meaning |
| --- | --- | --- |
| `compound_id` | yes | Name of the compound, or of the tracer in direct binding |
| `concentration` | yes | Concentration of the titrated species |
| `response` | yes | Measured signal |
| `experiment_id` | no | Identifies {term}`independent experiments <Independent experiment>` |

Rows with the same `compound_id`, `experiment_id` and `concentration` are
{term}`technical replicates <Technical replicate>`. They are averaged before
fitting, and each mean is weighted by the number of technical replicates. If
`experiment_id` is missing, all rows of a compound belong to one independent
experiment. Any other columns are kept in the table but not used.

Identifiers are converted to text, so `1` and `"1"` are the same compound.

## Technical replicates and independent experiments

The two kinds of replicates carry different information, and `bindcurve`
treats them differently.

**Technical replicates** are repeated measurements of the same sample within
one experiment, such as triplicate wells on one plate. They reflect the
precision of the measurement itself: pipetting and instrument noise. They share
an `experiment_id`, and `bindcurve` averages them before fitting.

**Independent experiments**, also called biological replicates, are complete
repeats of the experiment, prepared and measured separately: on another day,
with a fresh dilution series, a new protein preparation or a different cell
passage. They capture the variation between repeats of the whole experiment,
which is usually much larger than the scatter between technical replicates.
Each independent experiment gets its own `experiment_id` and is fitted
separately.

Only independent experiments count as $n$. The statistics in
{py:meth}`FitResults.summary() <bindcurve.FitResults.summary>` are computed
from the experiment-level estimates, and `N_fit` is the number of independent
experiments. Labeling technical replicates as separate experiments would
inflate $n$ and overstate the precision of the result. At least three
independent experiments are generally recommended before reporting a mean with
its spread.

## Concentrations

- Give concentrations on their original linear scale, not as logarithms.
- Use one unit for all concentrations and for all
  {term}`assay constants <Assay constant>`, such as `RT`, `LsT` and `Kds`.
  Fitted concentrations are reported in the same unit.
- Concentrations must be positive, so zero-concentration controls cannot be
  part of the table. Such controls are typically used to normalize the
  response, for example to percent of the uninhibited control, before the data
  are loaded.
- For the binding models, the concentration is that of the titrated species:
  total receptor in direct binding, total competitor in competitive binding.

## Loading data

From a pandas DataFrame:

```python
import bindcurve as bc

data = bc.DoseResponseData(table)
```

From a CSV file in long format:

```python
data = bc.DoseResponseData.from_csv("observations.csv")
```

Extra keyword arguments go to {py:func}`pandas.read_csv`, for example
`sep=";", decimal=","` for files that use a decimal comma.

### Wide format

In wide format, each row holds one concentration and the technical replicates
are in separate columns whose names start with `response_`:

```text
compound_id,experiment_id,concentration,response_1,response_2,response_3
inhibitor_a,exp_1,1.0,99.1,98.4,100.2
inhibitor_a,exp_1,3.0,97.0,96.1,97.9
```

```python
data = bc.DoseResponseData.from_csv("observations-wide.csv", format="wide")
```

Empty replicate cells are ignored, so compounds or concentrations can have
different numbers of replicates. A different column prefix can be given with
`prefix="..."`. A wide table can contain only the identifier, concentration and
response columns. The same conversion is available for DataFrames as
{py:meth}`DoseResponseData.from_wide() <bindcurve.DoseResponseData.from_wide>`.

### Other column names

If your columns have other names, rename them with pandas before creating the
data object:

```python
import pandas as pd

table = pd.read_csv("plate.csv").rename(
    columns={"Compound": "compound_id", "Conc_nM": "concentration", "mP": "response"}
)
data = bc.DoseResponseData(table)
```

## Checking and selecting data

`summary()` gives one row per compound with the number of independent
experiments (`N_exp`), the number of measured responses (`N_obs`) and the concentration
range. `compounds` lists the compound names, and `select()` returns a new data
object with a subset of compounds:

```python
data.summary()
data.compounds
subset = data.select(["inhibitor_a", "inhibitor_b"])
```

The validated table itself is available as `data.table`, a copy that can be
changed without affecting the data object.

## Validation errors

`DoseResponseData` raises a `ValueError` when:

- one of the required columns is missing, or the table is empty;
- `compound_id` or `experiment_id` has missing values;
- a concentration or response is missing, not numeric, or infinite;
- a concentration is zero or negative.

`select()` raises a `KeyError` for compound names that are not in the data.

## References

- Nature Communications, [Guidance on technical vs biological
  replicates](https://www.nature.com/documents/Biological_and_technical_replicates_guidelines.pdf).
- Heppner, D. E. et al. (2026), [Consistent quantitative reporting in medicinal
  chemistry](https://doi.org/10.1021/acs.jmedchem.6c01807). *J. Med. Chem.*
  69, 16115–16117.
