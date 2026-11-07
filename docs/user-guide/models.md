# Models

A model is chosen by name when fitting, for example `bc.fit(data, "ic50")`.
This page lists the models and what each needs. The equations and assumptions
are in the [theory](../theory/index.md) section.

## Available models

| Model | Experiment | x axis | Required fixed values | Fitted parameters |
| --- | --- | --- | --- | --- |
| `ic50` | Any inhibition dose-response | Inhibitor concentration | — | `ymin`, `ymax`, `IC50`, `hill_slope` |
| `dir_simple` | Direct binding | Receptor, taken as free | — | `ymin`, `ymax`, `Kds` |
| `dir_specific` | Direct binding | Total receptor | `LsT` | `ymin`, `ymax`, `Kds` |
| `dir_total` | Direct binding, nonspecific tracer binding | Total receptor | `LsT`, `Ns` | `ymin`, `ymax`, `Kds` |
| `comp_3st_specific` | Complete competition | Total competitor | `RT`, `LsT`, `Kds` | `ymin`, `ymax`, `Kd` |
| `comp_3st_total` | Complete competition, nonspecific competitor binding | Total competitor | `RT`, `LsT`, `Kds`, `N` | `ymin`, `ymax`, `Kd` |
| `comp_4st_specific` | Incomplete competition (ternary complex) | Total competitor | `RT`, `LsT`, `Kds`, `Kd3` | `ymin`, `ymax`, `Kd` |
| `comp_4st_total` | Incomplete competition, nonspecific competitor binding | Total competitor | `RT`, `LsT`, `Kds`, `Kd3`, `N` | `ymin`, `ymax`, `Kd` |

The required fixed values are {term}`assay constants <Assay constant>`: total
receptor `RT`, total tracer `LsT`, the tracer's dissociation constant `Kds`,
the dissociation constant `Kd3` of tracer from the competitor-bound receptor,
and the nonspecific-binding factors `Ns` and `N`. They must be passed in
`fit(..., fixed={...})`, in the same unit as the concentrations. The fitted
parameters can also be fixed when there is a reason to; see
[Fitting](fitting.md).

## Choosing a model

**IC₅₀ only.** If you need a potency that is comparable within one assay, and
do not know the receptor and tracer concentrations, fit `ic50`. An IC₅₀ depends
on the assay conditions, so it is not directly comparable between assays. See
the [logistic model](../theory/logistic.md).

**Tracer affinity.** To measure the $K_d$ of a labeled tracer, titrate receptor
into a fixed concentration of tracer and fit a direct-binding model:

- `dir_simple` when the tracer concentration is well below its $K_d$
  ($[L_T^*] \ll K_d^*$). The model then needs no assay constants.
- `dir_specific` otherwise. It accounts for receptor bound by tracer and needs
  `LsT`.
- `dir_total` when part of the tracer also binds nonspecifically, with a known
  factor `Ns`.

**Competitor affinity.** To measure the $K_d$ of an unlabeled compound, titrate
it into fixed receptor and tracer, and fit a competitive model with `RT`, `LsT`
and `Kds` from your assay. `Kds` usually comes from a direct-binding
experiment with the same tracer.

- `comp_3st_specific` when the competitor displaces the tracer completely.
- `comp_4st_specific` when receptor can bind tracer and competitor at the same
  time, so that the tracer is not fully displaced at high competitor
  concentration. It also needs `Kd3`, which can be measured by direct binding
  of the tracer to receptor saturated with competitor.
- The `_total` variants add nonspecific binding of the competitor with a known
  factor `N`.

**Converting an existing IC₅₀.** If an IC₅₀ from a competitive-binding assay
is already available, it can be converted to a $K_d$ without refitting; see
{ref}`converting-an-ic50-to-kd`.

The [equilibrium-binding theory](../theory/equilibrium.md) describes the
assumptions of each binding model.

## Model objects

Each model name corresponds to a model object, returned by
{py:func}`~bindcurve.get_model`. Its `parameters` list gives the name of each
parameter and whether it must be fixed:

```python
import bindcurve as bc

model = bc.get_model("comp_3st_specific")
for p in model.parameters:
    print(p.name, "fixed" if p.fixed else "fitted")
```

`model.evaluate(x, **params)` computes the response at concentrations `x` for
given parameter values, which is useful for simulating a planned experiment.
For binding models, `model.species(x, **params)` returns the equilibrium
concentrations of all species, such as free receptor `R` and the bound
fraction `Fbs`:

```python
model = bc.get_model("dir_specific")
species = model.species(
    [1.0, 10.0, 100.0], ymin=0.0, ymax=1.0, LsT=10.0, Kds=15.0
)
species["Fbs"]
```
