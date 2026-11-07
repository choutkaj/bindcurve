# Converting IC₅₀ to Kd

An $\mathrm{IC}_{50}$ depends on the assay, whereas $K_d$ is a property of the
competitor and the receptor. `bindcurve` provides three conversions from one to
the other. All three assume one-site competitive binding at equilibrium: tracer
and competitor bind the same site and cannot bind at the same time. The
competitor constant is written $K_d$ here; in pharmacology it is often called
the inhibition constant $K_i$.

The notation is:

| Symbol | Code | Meaning |
| --- | --- | --- |
| $[R_T]$ | `RT` | total receptor concentration |
| $[L_T^*]$ | `LsT` | total tracer concentration |
| $K_d^*$ | `Kds` | tracer dissociation constant |
| $K_d$ | `Kd` | competitor dissociation constant |
| $\mathrm{IC}_{50}$ | `IC50` | total competitor concentration that displaces half of the bound tracer |

A subscript 0 denotes the state without competitor and a subscript 50 the
state at $\mathrm{IC}_{50}$, so half displacement means
$[RL^*]_{50}=[RL^*]_0/2$.

An $\mathrm{IC}_{50}$ fitted with the [logistic model](logistic.md) matches
this definition when its top plateau is the competitor-free signal and its
bottom plateau is the signal with all specifically bound tracer displaced. When
receptor or tracer depletion is significant, the exact competition curve is not
a perfect logistic, so the fitted midpoint is itself an approximation. If the
assay constants are known, fitting the `comp_3st_specific` model gives $K_d$
directly.

The conversions do not apply to noncompetitive or irreversible inhibitors,
measurements taken before equilibrium, $\mathrm{IC}_{50}$ values from
functional assays, receptors with several kinds of sites, or the four-state
model of incomplete competition.

Some combinations of inputs are inconsistent with the binding model and give a
$K_d$ of zero or less. The conversion functions return NaN for these instead of
a meaningless affinity. A NaN $\mathrm{IC}_{50}$, such as one from a failed
fit, also gives NaN.

## Cheng-Prusoff approximation

Function: `cheng_prusoff`

`bindcurve` implements the usual form of the Cheng-Prusoff equation:

$$
K_d=\frac{\mathrm{IC}_{50}}
{1+\dfrac{[L_T^*]}{K_d^*}}.
$$

The equation assumes that the receptor binds only a small part of the tracer,
so that $[L^*]\approx[L_T^*]$, and only a small part of the competitor, so
that $\mathrm{IC}_{50}$ is close to the free competitor concentration. Both
hold when the receptor concentration is low, and the two exact corrections
below reduce to this equation in that limit.

## Munson-Rodbard correction

Function: `munson_rodbard`

Munson and Rodbard derived an exact form that also accounts for the receptor
concentration. Its input is the ratio of bound to free tracer without
competitor,

$$
y_0=\frac{[RL^*]_0}{[L^*]_0}.
$$

The competitor dissociation constant is then

$$
K_d=
\frac{\mathrm{IC}_{50}}
{1+
 \dfrac{[L_T^*](y_0+2)}{2K_d^*(y_0+1)}
 +y_0}
-K_d^*\frac{y_0}{y_0+2}.
$$

The formula uses total tracer, $[L_T^*]$. Replacing it with free tracer does
not give the exact result. The last term is subtracted: the original article
printed it with a plus sign, which the authors corrected in an erratum. As
$y_0\to0$, the formula reduces to the Cheng-Prusoff equation.

## Nikolovska-Coleska correction

Function: `coleska`

This exact correction computes the competitor-free state from $[R_T]$,
$[L_T^*]$ and $K_d^*$, so $y_0$ does not need to be measured. The calculation
goes step by step from the competitor-free state to half displacement.

Without competitor, free receptor solves the quadratic of the
[direct-binding model](equilibrium.md#direct-binding-with-depletion),

$$
[R]_0^2+a[R]_0+b=0,
\qquad
a=[L_T^*]+K_d^*-[R_T],
\qquad
b=-K_d^*[R_T],
$$

whose non-negative root is

$$
[R]_0=\frac{-a+\sqrt{a^2-4b}}{2}.
$$

Free and bound tracer follow from the tracer equilibrium and the receptor
balance:

$$
[L^*]_0=\frac{[L_T^*]}{1+[R]_0/K_d^*},
\qquad
[RL^*]_0=\frac{[R_T]}{1+K_d^*/[L^*]_0}.
$$

At $\mathrm{IC}_{50}$, half of this complex remains, and the rest of the
tracer is free:

$$
[RL^*]_{50}=\frac{[RL^*]_0}{2},
\qquad
[L^*]_{50}=[L_T^*]-[RL^*]_{50}.
$$

The tracer equilibrium then gives free receptor,

$$
[R]_{50}=K_d^*\frac{[RL^*]_{50}}{[L^*]_{50}},
$$

and the receptor balance gives the competitor complex,

$$
[RL]_{50}=[R_T]-[R]_{50}-[RL^*]_{50}.
$$

The free competitor is what remains of the total,

$$
[L]_{50}=\mathrm{IC}_{50}-[RL]_{50},
$$

and the competitor equilibrium gives

$$
K_d=\frac{[R]_{50}[L]_{50}}{[RL]_{50}}.
$$

The implementation uses the equivalent form published by
Nikolovska-Coleska et al.,

$$
K_d=\frac{[L]_{50}}
{1+\dfrac{[L^*]_{50}+[R]_0}{K_d^*}}.
$$

The Munson-Rodbard and Nikolovska-Coleska corrections give the same $K_d$ when
their inputs describe the same assay. Use Munson-Rodbard when you have
measured $y_0$, and Nikolovska-Coleska when you know $[R_T]$.

## References

- Cheng, Y. & Prusoff, W. H. (1973), [Relationship between the inhibition
  constant and the concentration of inhibitor which causes 50 per cent
  inhibition](https://doi.org/10.1016/0006-2952(73)90196-2).
- Munson, P. J. & Rodbard, D. (1988), [An exact correction to the
  Cheng-Prusoff correction](https://doi.org/10.3109/10799898809049010), with
  the [published erratum](https://doi.org/10.3109/10799898909066075).
- Nikolovska-Coleska, Z. et al. (2004), [Development and optimization of a
  binding assay for the XIAP BIR3 domain using fluorescence
  polarization](https://doi.org/10.1016/j.ab.2004.05.055).
