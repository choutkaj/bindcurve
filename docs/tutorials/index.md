# Tutorials

The tutorials are complete analyses of synthetic data sets, one for each type
of experiment. Each data set has three independent experiments with technical
triplicates, and concentrations in nM. The data were generated with
[`make_data.py`](https://github.com/choutkaj/bindcurve/blob/main/docs/tutorials/data/make_data.py),
which lists the true parameter values.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} IC₅₀ analysis
:link: ic50-analysis
:link-type: doc

IC₅₀ values of two enzyme inhibitors.
:::

:::{grid-item-card} Direct binding
:link: direct-binding
:link-type: doc

Tracer Kd values from FP titrations of receptor.
:::

:::{grid-item-card} Competitive binding
:link: competitive-binding
:link-type: doc

Competitor Kd values from FP competition, complete and incomplete.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

ic50-analysis
direct-binding
competitive-binding
```
