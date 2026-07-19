# Logistic model

The `ic50` model is an empirical four-parameter inhibitory dose-response
model. It describes a monotonic decrease from an upper response to a lower
response as the concentration increases:

$$
y(x) = y_{\min}
     + \frac{A}{1 + \left(\dfrac{x}{\mathrm{IC}_{50}}\right)^h}.
$$

The parameters are:

- $x \ge 0$: the **raw**, untransformed concentration;
- $y_{\min}$: the response approached at high concentration;
- $A > 0$: the response amplitude;
- $\mathrm{IC}_{50} > 0$: the concentration at the midpoint;
- $h > 0$: the Hill slope, exposed as `hill_slope`.

The upper response is $y_{\max}=y_{\min}+A$. At the fitted midpoint,

$$
y(\mathrm{IC}_{50}) = y_{\min} + \frac{A}{2}.
$$

The model is therefore identifiable in one orientation: `amplitude` and
`hill_slope` are constrained to be positive, and increasing concentration
always decreases the predicted response. It does not represent an activating
curve by reversing the sign of the slope.

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
