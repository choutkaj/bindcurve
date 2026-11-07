# Logistic model

The `ic50` model is the empirical four-parameter dose-response curve used to
estimate an IC₅₀. In an inhibition assay, the response falls from an upper
plateau to a lower plateau as the concentration increases:

$$
y(x) = y_{\min}
     + \frac{y_{\max}-y_{\min}}
       {1 + \left(\dfrac{x}{\mathrm{IC}_{50}}\right)^h},
$$

where

- $x \ge 0$ is the concentration, on its original linear scale;
- $y_{\max}$ is the response at zero concentration;
- $y_{\min}$ is the response at very high concentration;
- $\mathrm{IC}_{50} > 0$ is the concentration at the midpoint of the curve;
- $h > 0$ is the Hill slope, called `hill_slope` in the code.

At $x=\mathrm{IC}_{50}$ the response is halfway between the two plateaus:

$$
y(\mathrm{IC}_{50}) = \frac{y_{\min}+y_{\max}}{2}.
$$

The plateaus are not constrained. If the data rise with concentration, the fit
returns $y_{\min} > y_{\max}$: `ymin` is still the high-concentration response
and `ymax` the zero-concentration response, even though the names then no
longer match the smaller and larger values.

This is the same curve as GraphPad Prism's
[\[Inhibitor\] vs. response, variable slope](https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_dr_inhibit_variable_2.htm)
equation, with `ymax` as **Top** and `ymin` as **Bottom**. Prism gives an
inhibition curve a negative `HillSlope`; `bindcurve` uses the positive slope
$h = -\mathtt{HillSlope}$.

## Concentrations and logarithms

Give `bindcurve` concentrations on their original linear scale, not as
$\log_{10}(x)$. The model depends only on the ratio $x/\mathrm{IC}_{50}$, so
any concentration unit works, as long as it is used consistently.

Internally, the optimizer works with $\log_{10}$ of every concentration-like
parameter, including $\mathrm{IC}_{50}$. This keeps the parameters positive
and makes the fit better conditioned. It does not change the model: the fitted
$\mathrm{IC}_{50}$ is reported in the unit of the input concentrations.

Results from several experiments are combined on the $\log_{10}$ scale. In the
summary table, `logIC50` is the mean of $\log_{10}(\mathrm{IC}_{50})$ over
experiments, and `IC50` is $10^{\mathtt{logIC50}}$, i.e. the geometric mean.
Both are in the input unit.

```{important}
$\mathrm{pIC}_{50}$ is defined as $-\log_{10}(\mathrm{IC}_{50}/1\ \mathrm{M})$,
so $\mathrm{pIC}_{50} = -\mathtt{logIC50}$ only when the concentrations are in
molar. If they are in µM, $\mathrm{pIC}_{50} = 6 - \mathtt{logIC50}$; for
example, $\mathrm{IC}_{50} = 1\ \mathrm{\mu M}$ gives $\mathrm{pIC}_{50} = 6$.
```

## Interpretation

$\mathrm{IC}_{50}$ describes the curve under the conditions of a particular
assay. It is generally not a dissociation constant. How it relates to the
competitor's $K_d$ depends on the binding mechanism, the tracer and receptor
concentrations, ligand depletion, and whether the system reached equilibrium.
For one-site competitive binding at equilibrium, see
[Converting IC₅₀ to Kd](conversions.md).
