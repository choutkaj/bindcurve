# BindCurve data formats

BindCurve stores observations in a validated long-form table and supports `long`
and `wide` layouts through `from_dataframe()`, `from_csv()`, `from_json()`,
`to_dataframe()`, `to_csv()`, and `to_json()`.

## Long format

Each row is one technical replicate observation:

```csv
compound_id,experiment_id,concentration,replicate_id,response
Cmpd_1,exp_1,0.001,rep_1,98.1
Cmpd_1,exp_1,0.001,rep_2,97.5
Cmpd_1,exp_1,0.003,rep_1,94.2
Cmpd_1,exp_2,0.001,rep_1,97.9
```

Required columns are `compound_id`, `concentration`, and `response`. Missing
`experiment_id` defaults to `experiment_1`. Missing `replicate_id` is generated
within each compound, experiment, and concentration group.

Identifiers must be nonmissing and nonblank. Concentrations must be finite and
positive; responses must be finite. Duplicate observation identities are rejected.
Additional observation metadata columns are preserved in long format.

Known observation uncertainty can be supplied as either `sigma` (standard
deviation) or `weight` (reciprocal standard deviation), never both. Values must be
finite and positive. These are known measurement uncertainties, not empirical
replicate SD or SEM.

```python
import bindcurve as bc

data = bc.DoseResponseData.from_csv(
    "observations.csv",
    metadata={"concentration_unit": "uM", "response_unit": "percent"},
)
```

BindCurve performs no unit conversion. Metadata annotates the numerical scale;
all concentration-like inputs must already use a consistent scale.

## Wide format

Each row represents one compound, experiment, and concentration; response columns
contain technical replicates:

```csv
compound_id,experiment_id,concentration,response_1,response_2,response_3
Cmpd_1,exp_1,0.001,98.1,97.5,99.0
Cmpd_1,exp_1,0.003,94.2,95.0,93.7
Cmpd_1,exp_2,0.001,97.9,98.3,98.6
```

```python
data = bc.DoseResponseData.from_csv("observations-wide.csv", format="wide")
```

Columns beginning with `response_` are discovered by default. Use `replicate_cols`
or `replicate_prefix` to customize discovery. Missing response cells are omitted.
Non-replicate metadata and uncertainty columns are unsupported in wide format;
use long format to retain them.

Wide export requires replicate identifiers consisting of the selected prefix
(default `response_`) followed by an integer. Data imported from the default wide
layout already meets this requirement. Automatically generated long-format IDs
use `replicate_`, so exporting those IDs requires `replicate_prefix="replicate_"`.
Arbitrary named replicate IDs cannot be represented by wide export.

## Column mappings and serialization

```python
data = bc.DoseResponseData.from_dataframe(
    observations,
    compound_col="compound",
    concentration_col="dose",
    response_col="signal",
)

long_table = data.to_dataframe()
data.to_csv("normalized.csv")
json_text = data.to_json()
restored = bc.DoseResponseData.from_json(json_text)
```

Use the same column mappings when importing an export with custom column names.
Output names must be unique, including retained metadata and wide response
columns. Colliding mappings raise an error before serialization.

CSV stores the table only. JSON emitted by `to_json()` contains `format`,
`metadata`, and `table`, and preserves dataset metadata. `from_json()` accepts
JSON text or a file path, and also accepts a bare table payload. An explicitly
requested format must agree with the JSON envelope's format.

Public table and dataset-metadata access returns isolated copies. Filtering
preserves row order and metadata. Concatenation requires matching dataset
metadata and nonoverlapping compound/experiment identities.
