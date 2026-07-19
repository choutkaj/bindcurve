# Converting IC₅₀ to $K_d$

An IC₅₀ is an assay-dependent midpoint, whereas $K_d$ is an equilibrium
dissociation constant. `bindcurve` implements three conversions for the
specific case of one-site, mutually exclusive, competitive equilibrium between
a labeled tracer and an unlabeled competitor.

The notation is:

| Symbol | Code | Meaning |
| --- | --- | --- |
| $[R_T]$ | `RT` | total receptor concentration |
| $[L_T^*]$ | `LsT` | total tracer concentration |
| $K_d^*$ | `Kds` | tracer dissociation constant |
| $K_d$ | `Kd` | competitor dissociation constant |
| $\mathrm{IC}_{50}$ | `IC50` | total competitor giving 50% tracer displacement |

Here, 50% displacement means
$[RL^*]_{50}=[RL^*]_0/2$, where the subscript 0 denotes the
competitor-free state. These conversions do not apply to noncompetitive or
irreversible mechanisms, nonequilibrium measurements, functional-response
IC₅₀ values, heterogeneous sites, or the four-state incomplete-competition
model.

## Cheng-Prusoff approximation

Function: `cheng_prusoff`

The familiar conversion used by the API is

$$
K_d=\frac{\mathrm{IC}_{50}}
{1+\dfrac{[L_T^*]}{K_d^*}}.
$$

The classical relationship is written in terms of free tracer. Substitution of
total tracer is accurate when tracer depletion by receptor is negligible, so
$[L^*]\approx[L_T^*]$, and bound competitor contributes negligibly to its total
concentration. In `bindcurve`, this approximation is the low-receptor limit of
the finite-concentration conversions below.

## Munson-Rodbard finite-concentration correction

Function: `cheng_prusoff_corrected`

Let

$$
y_0=\frac{[RL^*]_0}{[L^*]_0}
$$

be the bound-to-free tracer ratio before competitor is added. Under the same
one-site competitive-equilibrium model, the exact finite-concentration
correction is

$$
K_d=
\frac{\mathrm{IC}_{50}}
{1+
 \dfrac{[L_T^*](y_0+2)}{2K_d^*(y_0+1)}
 +y_0}
-K_d^*\frac{y_0}{y_0+2}.
$$

The second term is **subtracted**. The original article printed a plus sign;
the authors' erratum corrected it to a minus sign. When $y_0\to0$, this
expression reduces to the Cheng-Prusoff approximation.

Some combinations of $\mathrm{IC}_{50}$, $[L_T^*]$, $K_d^*$, and $y_0$ are
incompatible with the assumed equilibrium and produce a non-positive result.
`bindcurve` rejects such inputs rather than reporting them as affinities.

## Nikolovska-Coleska finite-concentration correction

Function: `coleska`

This form derives the competitor-free state from $[R_T]$, $[L_T^*]$, and
$K_d^*$ instead of requiring $y_0$.

Before competitor is added, free receptor obeys

$$
[R_0]^2+a[R_0]+b=0,
$$

where

$$
a=[L_T^*]+K_d^*-[R_T],
\qquad
b=-K_d^*[R_T].
$$

The physical root is

$$
[R_0]=\frac{-a+\sqrt{a^2-4b}}{2}.
$$

The remaining competitor-free quantities are

$$
[L_0^*]=\frac{[L_T^*]}{1+[R_0]/K_d^*},
\qquad
[RL_0^*]=\frac{[R_T]}{1+K_d^*/[L_0^*]}.
$$

At 50% displacement,

$$
[RL_{50}^*]=\frac{[RL_0^*]}{2},
\qquad
[L_{50}^*]=[L_T^*]-[RL_{50}^*],
$$

and tracer equilibrium gives

$$
[R_{50}]=K_d^*\frac{[RL_{50}^*]}{[L_{50}^*]}.
$$

Receptor conservation then requires

$$
[RL_{50}]=[R_T]-[R_{50}]-[RL_{50}^*].
$$

Both terms on the right are subtracted. The free competitor concentration is

$$
[L_{50}]=\mathrm{IC}_{50}-[RL_{50}],
$$

and the competitor dissociation constant follows directly from its equilibrium:

$$
K_d=\frac{[R_{50}][L_{50}]}{[RL_{50}]}.
$$

An algebraically equivalent final expression used by the implementation is

$$
K_d=\frac{[L_{50}]}
{1+\dfrac{[L_{50}^*]+[R_0]}{K_d^*}}.
$$

Under consistent inputs, the Munson-Rodbard and Nikolovska-Coleska corrections
recover the same one-site equilibrium $K_d$. The former requires $y_0$; the
latter derives the initial state from total receptor and tracer concentrations.

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
