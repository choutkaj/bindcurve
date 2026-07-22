# Logistic model

The `ic50` model is an empirical four-parameter inhibitory dose-response
model. It describes a monotonic decrease from an upper response to a lower
response as the concentration increases:

$$
y(x) = y_{\min}
     + \frac{y_{\max}-y_{\min}}
       {1 + \left(\dfrac{x}{\mathrm{IC}_{50}}\right)^h}.
$$

The parameters are:

- $x \ge 0$: the **raw**, untransformed concentration;
- $y_{\min}$: the response approached at high concentration;
- $y_{\max}$: the response approached at zero concentration;
- $\mathrm{IC}_{50} > 0$: the concentration at the midpoint;
- $h > 0$: the Hill slope, exposed as `hill_slope`.

$$
y(\mathrm{IC}_{50}) = \frac{y_{\min}+y_{\max}}{2}.
$$

This corresponds to GraphPad Prism's
[variable-slope inhibitory dose-response equation](https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_dr_inhibit_variable_2.htm):
`ymax` is **Top** and `ymin` is **Bottom**. `bindcurve` writes the inhibitory
Hill slope as the positive quantity $h$; it is the negative of Prism's signed
`HillSlope`. The curve decreases when $y_{\max}>y_{\min}$.

## Concentration and logarithms

Concentrations supplied to `bindcurve` must remain on their original linear
scale. The model is evaluated from the dimensionless ratio
$x/\mathrm{IC}_{50}$; it does not accept $\log_{10}(x)$ as its input axis.

For numerical stability and to enforce positivity, `bindcurve` optimizes
$\mathrm{IC}_{50}$ internally on a base-10 logarithmic coordinate. This is an
optimizer detail, not a separate `logIC50` model. The public fitted value is
returned on the same concentration scale as the input data, and

$$
\log\mathrm{IC}_{50}=\log_{10}(\mathrm{IC}_{50}).
$$

```{important}
The conventional quantity $\mathrm{pIC}_{50}$ is
$-\log_{10}(\mathrm{IC}_{50}/1\ \mathrm{M})$. The numerical concentration must
therefore be expressed in molar units before taking the negative logarithm.
For example, $1\ \mathrm{\mu M}$ corresponds to $\mathrm{pIC}_{50}=6$, not 0.
```

## Interpretation

$\mathrm{IC}_{50}$ is a curve midpoint under the conditions of the assay. It
is not generally a thermodynamic dissociation constant. Its relationship to a
competitor $K_d$ depends on the binding mechanism, tracer concentration,
receptor concentration, depletion, and whether equilibrium was reached. See
the [IC₅₀-to-$K_d$ conversions](conversions.md) for the one-site competitive
equilibrium case.
