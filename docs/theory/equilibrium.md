# Equilibrium binding models

`bindcurve` includes mass-balance models for direct and competitive binding.
Unlike the empirical IC₅₀ model, these models assign physical meaning to the
concentrations and fitted dissociation constants.

The notation used below is:

| Symbol | Code | Meaning |
| --- | --- | --- |
| $R$, $R_T$ | `R`, `RT` | free and total receptor |
| $L^*$, $L_T^*$ | `Ls`, `LsT` | free and total labeled ligand (tracer) |
| $L$, $L_T$ | `L`, `LT` | free and total unlabeled ligand (competitor) |
| $RL^*$ | `RLs` | receptor-tracer complex |
| $RL$ | `RL` | receptor-competitor complex |
| $RLL^*$ | `RLLs` | ternary receptor-competitor-tracer complex |
| $K_d^*$ | `Kds` | tracer dissociation constant |
| $K_d$ | `Kd` | competitor dissociation constant |
| $K_{d3}$ | `Kd3` | tracer dissociation constant from $RL$ |
| $N^*$, $N$ | `Ns`, `N` | linear nonspecific-binding factors |

All concentration-like quantities must use the same numerical unit. The
models assume equilibrium, a single receptor population, ideal dilute-solution
behavior, and the stated binding stoichiometry.

The physical tracer-bound fraction, $F_b^*$, is mapped to the measured response
with

$$
y = y_{\min} + (y_{\max}-y_{\min})F_b^*.
$$

This assumes that the measured response is affine in the modeled bound
fraction. The fitted $y_{\min}$ and $y_{\max}$ correspond to hypothetical
$F_b^*=0$ and $F_b^*=1$ states; for finite-concentration competitive models
they need not equal the smallest and largest responses actually observed in a
titration.

```{note}
For fluorescence-polarization assays, unequal fluorescence intensities of the
free and bound tracer can make the relationship between anisotropy and bound
fraction nonlinear. The present models do not include that additional
photophysical correction.
```

## Direct binding

The direct system contains receptor and tracer,

$$
R + L^* \rightleftharpoons RL^*,
\qquad
K_d^* = \frac{[R][L^*]}{[RL^*]}.
$$

For specific binding,

$$
F_b^* = \frac{[RL^*]}{[L_T^*]}
      = \frac{[R]}{K_d^*+[R]}.
$$

### Simplified direct binding

Model: `dir_simple`

The simplified model treats the concentration axis as free receptor:

$$
F_b^* = \frac{[R]}{K_d^*+[R]}.
$$

When experimental input is total receptor, this amounts to the approximation
$[R]\approx[R_T]$. It is appropriate only when tracer depletion and receptor
occupancy do not materially separate free from total receptor. Its midpoint is
$[R]=K_d^*$.

### Depletion-aware direct binding

Model: `dir_specific`

The exact one-site mass balances are

$$
[L_T^*]=[L^*]+[RL^*],
\qquad
[R_T]=[R]+[RL^*].
$$

Eliminating the other species gives a quadratic in free receptor,

$$
[R]^2+a[R]+b=0,
$$

with

$$
a=K_d^*+[L_T^*]-[R_T],
\qquad
b=-K_d^*[R_T].
$$

The physical solution is the non-negative root

$$
[R]=\frac{-a+\sqrt{a^2-4b}}{2}.
$$

`bindcurve` evaluates this root in a rationalized form when needed to avoid
loss of precision. The complex concentration and observable are then

$$
[RL^*]=[R_T]-[R],
\qquad
F_b^*=\frac{[RL^*]}{[L_T^*]}.
$$

### Direct binding with nonspecific tracer immobilization

Model: `dir_total`

Nonspecific tracer immobilization is represented as a nonsaturable amount
proportional to free tracer,

$$
[L^*]_{\mathrm{NS}}=N^*[L^*],
\qquad N^*\ge0.
$$

The tracer balance becomes

$$
[L_T^*]=(1+N^*)[L^*]+[RL^*],
$$

while the receptor balance remains

$$
[R_T]=[R]+[RL^*].
$$

Free receptor again satisfies a quadratic, now with

$$
a=(1+N^*)K_d^*+[L_T^*]-[R_T],
\qquad
b=-(1+N^*)K_d^*[R_T].
$$

The physical total bound fraction is

$$
F_{b,\mathrm{total}}^*
=\frac{[RL^*]+N^*[L^*]}{[L_T^*]}.
$$

At zero receptor its nonspecific baseline is
$F_0=N^*/(1+N^*)$. Because `ymin` represents that fitted baseline,
`dir_total` maps the baseline-normalized fraction

$$
\frac{F_{b,\mathrm{total}}^*-F_0}{1-F_0}
=\frac{[RL^*]}{[L_T^*]}
$$

to the response. The component API exposes both the specific and total bound
fractions. The factor $(1+N^*)$ changes the mass balance and the apparent
location of the curve; it does **not** mean that the microscopic $K_d^*$ has
changed or that total binding is simply $(1+N^*)$ times specific binding.

## Complete competitive binding: three states

Models: `comp_3st_specific`, `comp_3st_total`

In the complete-competition model, tracer and competitor bind mutually
exclusively:

$$
R+L^*\rightleftharpoons RL^*,
\qquad
R+L\rightleftharpoons RL.
$$

The independent equilibria are

$$
K_d^*=\frac{[R][L^*]}{[RL^*]},
\qquad
K_d=\frac{[R][L]}{[RL]}.
$$

For specific binding, conservation of mass requires

$$
\begin{aligned}
[L_T^*]&=[L^*]+[RL^*],\\
[L_T]&=[L]+[RL],\\
[R_T]&=[R]+[RL^*]+[RL].
\end{aligned}
$$

Consequently,

$$
[RL^*]=\frac{[L_T^*][R]}{K_d^*+[R]},
\qquad
[RL]=\frac{[L_T][R]}{K_d+[R]},
$$

and the measured tracer fraction is

$$
F_b^*=\frac{[RL^*]}{[L_T^*]}
     =\frac{[R]}{K_d^*+[R]}.
$$

In particular, the denominator contains the **tracer** dissociation constant
$K_d^*$, not the competitor dissociation constant $K_d$.

Substitution into the receptor balance gives

$$
[R]^3+a[R]^2+b[R]+c=0,
$$

where

$$
\begin{aligned}
a={}&K_d^*+K_d+[L_T^*]+[L_T]-[R_T],\\
b={}&K_d^*([L_T]-[R_T])
      +K_d([L_T^*]-[R_T])+K_d^*K_d,\\
c={}&-K_d^*K_d[R_T].
\end{aligned}
$$

Although this cubic has a closed-form physical root, `bindcurve` solves the
monotonic receptor mass balance directly on $0\le[R]\le[R_T]$. This avoids the
branch selection and numerical cancellation problems of the explicit cubic
formula.

For `comp_3st_total`, nonspecific competitor immobilization is

$$
[L]_{\mathrm{NS}}=N[L]
$$

and the competitor balance becomes

$$
[L_T]=(1+N)[L]+[RL].
$$

Within the receptor mass balance this is algebraically equivalent to replacing
$K_d$ by $(1+N)K_d$. The parameter $K_d$ nevertheless remains the microscopic
specific dissociation constant; $(1+N)K_d$ is an effective term created by the
additional ligand sink.

## Incomplete competitive binding: four states

Models: `comp_4st_specific`, `comp_4st_total`

The four-state model allows tracer and competitor to occupy the receptor
simultaneously:

$$
\begin{aligned}
R+L^*&\rightleftharpoons RL^*,\\
R+L&\rightleftharpoons RL,\\
RL+L^*&\rightleftharpoons RLL^*,\\
RL^*+L&\rightleftharpoons RLL^*.
\end{aligned}
$$

`bindcurve` parameterizes the first three equilibria as

$$
K_d^*=\frac{[R][L^*]}{[RL^*]},
\qquad
K_d=\frac{[R][L]}{[RL]},
\qquad
K_{d3}=\frac{[RL][L^*]}{[RLL^*]}.
$$

The fourth constant is not an independent parameter. Thermodynamic consistency
around the binding cycle requires

$$
K_{d4}=\frac{[RL^*][L]}{[RLL^*]}
       =\frac{K_dK_{d3}}{K_d^*}.
$$

The three mass balances are

$$
\begin{aligned}
[L_T^*]&=[L^*]+[RL^*]+[RLL^*],\\
[L_T]&=[L]+[RL]+[RLL^*],\\
[R_T]&=[R]+[RL^*]+[RL]+[RLL^*].
\end{aligned}
$$

Given the free concentrations,

$$
\begin{aligned}
[RL^*]&=\frac{[R][L^*]}{K_d^*},\\
[RL]&=\frac{[R][L]}{K_d},\\
[RLL^*]&=\frac{[R][L][L^*]}{K_dK_{d3}}.
\end{aligned}
$$

The observable must count tracer in both receptor-bound species:

$$
F_b^*=\frac{[RL^*]+[RLL^*]}{[L_T^*]}.
$$

Eliminating the free ligands generally produces a quintic polynomial in true
free receptor. `bindcurve` finds candidate roots in the physical interval
$0\le[R]\le[R_T]$, reconstructs all species to enforce non-negativity and the
mass balances, and falls back to direct numerical solution of the receptor
balance when necessary.

The limiting behavior gives useful checks on the model:

- At $[L_T]=0$, it reduces to direct tracer binding with $K_d^*$.
- At very large $[L_T]$, receptor is predominantly competitor-bound and tracer
  binding approaches a direct-binding curve governed by $K_{d3}$:

  $$
  \lim_{[L_T]\to\infty}F_b^*
  =\frac{K_{d3}+[L_T^*]+[R_T]
  -\sqrt{(K_{d3}+[L_T^*]+[R_T])^2-4[L_T^*][R_T]}}
  {2[L_T^*]}.
  $$

- If $K_{d3}=K_d^*$, competitor occupancy does not alter tracer affinity and
  $F_b^*$ is independent of competitor concentration.
- If $K_{d3}\to\infty$, the ternary state is suppressed and the model tends
  toward complete three-state competition.

Thus $K_{d3}<K_d^*$ describes favorable coupling of tracer binding to the
competitor-bound receptor, whereas $K_{d3}>K_d^*$ describes unfavorable
coupling. A finite $K_{d3}$ can produce a nonzero high-competitor plateau.

For `comp_4st_total`, nonspecific competitor immobilization adds
$N[L]$ to the competitor balance:

$$
[L_T]=(1+N)[L]+[RL]+[RLL^*].
$$

As in the three-state total model, the observable equilibrium can be solved
with the effective term $(1+N)K_d$, while the reported $K_d$ remains the
microscopic competitor dissociation constant.

## References

- Wang, Z.-X. (1995), [An exact mathematical expression for describing
  competitive binding of two different ligands to a protein
  molecule](https://doi.org/10.1016/0014-5793(95)00062-E).
- Roehrl, M. H. A., Wang, J. Y. & Wagner, G. (2004), [A general framework for
  development and data analysis of competitive high-throughput screens for
  small-molecule inhibitors of protein-protein interactions by fluorescence
  polarization](https://doi.org/10.1021/bi048233g).
