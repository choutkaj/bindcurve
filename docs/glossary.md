# Glossary

```{glossary}
Assay constant
  A quantity set by the experimental design rather than fitted, such as total
  receptor (`RT`), total tracer (`LsT`) or the tracer's dissociation constant
  (`Kds`). Binding models require their assay constants in
  `fit(..., fixed={...})`.

Biological replicate
  See {term}`independent experiment`.

Bound fraction
  The fraction of tracer that is bound to receptor, $F_b^*$. The equilibrium
  models compute it and map it linearly to the response.

Competitive binding
  An assay in which unlabeled {term}`competitor` is titrated into a fixed
  mixture of receptor and {term}`tracer`. The competitor's affinity is
  inferred from how much tracer it displaces.

Competitor
  The unlabeled ligand in a competitive-binding assay, usually the compound
  being characterized.

Concentration-like parameter
  A parameter with concentration units, such as `IC50`, `Kd` or `RT`.
  `bindcurve` fits these on a $\log_{10}$ scale and averages them across
  independent experiments as {term}`geometric means <Geometric mean>`.

Depletion
  A significant part of a species being consumed by binding, so that its free
  concentration differs noticeably from its total concentration. Models that
  account for depletion use total concentrations and solve the mass balances.

Direct binding
  An assay in which receptor is titrated into a fixed concentration of
  {term}`tracer`, to measure the tracer's own dissociation constant.

Dissociation constant
  $K_d = [R][L]/[RL]$ at equilibrium. It equals the free-ligand concentration
  at which half of the receptor sites are occupied. A lower $K_d$ means
  tighter binding. In `bindcurve`, `Kds` is the tracer's constant and `Kd` the
  competitor's.

Fixed parameter
  A parameter held at a given value during fitting instead of being
  estimated. Set with `fit(..., fixed={...})`.

Geometric mean
  The mean of the logarithms, transformed back: $10^{\overline{\log_{10} x}}$.
  Used for concentration-like parameters such as IC₅₀ and $K_d$, whose
  values from independent experiments are approximately log-normally
  distributed.

Hill slope
  The steepness parameter $h$ of the {term}`logistic model`. $h = 1$ is the
  steepness expected for simple one-site binding; larger values give steeper
  curves.

IC₅₀
  The concentration of an inhibitor or competitor that brings the response
  halfway between the two {term}`plateaus <Plateau>`. It depends on the assay
  conditions and is not in general a {term}`dissociation constant`.

Independent experiment
  A complete repeat of a dose-response experiment, prepared and measured
  separately, for example on a different day with a fresh dilution series or
  protein preparation. Also called a biological or experimental replicate.
  Independent experiments capture the variation between repeats of the whole
  experiment, which is usually much larger than the scatter between
  {term}`technical replicates <Technical replicate>`. They are identified by
  the `experiment_id` column, fitted separately, and form the $n$ of the
  summary statistics.

Logistic model
  The empirical four-parameter dose-response curve (`ic50` model), with two
  plateaus, a midpoint and a Hill slope.

Nonspecific binding
  Binding of a ligand to something other than the receptor site, such as the
  plate or other proteins. The `_total` models treat it as proportional to the
  free ligand concentration, with factor `Ns` (tracer) or `N` (competitor).

pIC₅₀
  $-\log_{10}(\mathrm{IC}_{50})$ with the IC₅₀ expressed in mol/L. An IC₅₀ of
  1 µM corresponds to a pIC₅₀ of 6.

Plateau
  The response that a curve approaches at zero or at very high concentration.
  Every model has the two plateau parameters `ymin` and `ymax`.

Receptor
  The binding partner whose sites tracer and competitor occupy, for example a
  protein. Its total concentration is `RT`.

Replicate mean
  The average of the {term}`technical replicates <Technical replicate>` at one
  concentration within one independent experiment. Fits use replicate means,
  not single wells.

Residual
  The difference between a measured replicate mean and the fitted curve at the
  same concentration.

Technical replicate
  A repeated measurement of the same sample within one
  {term}`independent experiment`, such as duplicate or triplicate wells on one
  plate. Technical replicates reflect the precision of the measurement itself
  (pipetting, instrument noise), not the variation between experiments. Rows
  with the same compound, experiment and concentration are technical
  replicates; they are averaged before fitting and do not count towards $n$.

Ternary complex
  A receptor bound to tracer and competitor at the same time ($RLL^*$),
  allowed only in the four-state models.

Tracer
  A labeled ligand whose binding produces the measured signal, for example a
  fluorescent probe. Its total concentration is `LsT`.
```
