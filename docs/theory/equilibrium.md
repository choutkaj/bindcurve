# Equilibrium binding models

The equilibrium models describe direct and competitive binding with explicit
mass balances. Unlike the empirical IC₅₀ model, their concentrations and fitted
dissociation constants have a physical meaning.

The notation is:

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
| $K_{d3}$ | `Kd3` | dissociation constant of tracer from $RL$ |
| $N^*$, $N$ | `Ns`, `N` | nonspecific-binding factors for tracer and competitor |

All concentrations and dissociation constants must be in the same unit. The
models assume that the system is at equilibrium, that there is a single
population of receptor sites, and that each complex has the stoichiometry
shown below.

Model names ending in `_specific` include only binding to the receptor. Names
ending in `_total` also include nonspecific binding of the titrated ligand,
modeled as a sink that never saturates.

## From bound fraction to response

Every equilibrium model computes $F_b^*$, the fraction of tracer bound to
receptor, and maps it to the measured response with

$$
y = y_{\min} + (y_{\max}-y_{\min})F_b^*.
$$

This assumes that the signal is a linear function of the bound fraction. The
fitted $y_{\min}$ and $y_{\max}$ are the responses of fully free
($F_b^*=0$) and fully bound ($F_b^*=1$) tracer. A competition titration
often does not reach one or both of these states, so $y_{\min}$ and $y_{\max}$
need not match the lowest and highest responses in the data.

```{note}
In fluorescence polarization, a change in brightness when the tracer binds
makes the measured anisotropy a nonlinear function of the bound fraction. The
models do not correct for this.
```

## Direct binding

In a direct-binding experiment, receptor is titrated into a fixed amount of
tracer:

$$
R + L^* \rightleftharpoons RL^*,
\qquad
K_d^* = \frac{[R][L^*]}{[RL^*]}.
$$

Because $[L_T^*]=[L^*]+[RL^*]$, the bound fraction depends only on free
receptor:

$$
F_b^* = \frac{[RL^*]}{[L_T^*]}
      = \frac{[R]}{K_d^*+[R]}.
$$

The three direct models differ in how they obtain $[R]$ from the receptor
concentration on the x axis.

### Simplified direct binding

Model: `dir_simple`

The simplified model takes the x axis to be free receptor and evaluates the
expression above directly. Its midpoint is at $[R]=K_d^*$.

In a real titration the x axis is usually total receptor, so the model
assumes $[R]\approx[R_T]$. Since $[R_T]=[R](1+[L^*]/K_d^*)$ and
$[L^*]\le[L_T^*]$,

$$
\frac{[R]}{[R_T]} = \frac{1}{1+[L^*]/K_d^*}
\ge \frac{1}{1+[L_T^*]/K_d^*}.
$$

The approximation therefore holds over the whole titration when
$[L_T^*]\ll K_d^*$, so that tracer occupies only a small fraction of the
receptor. A large fraction of *tracer* being bound is not a problem in itself.

### Direct binding with depletion

Model: `dir_specific`

This model makes no approximation. It solves the two mass balances

$$
[L_T^*]=[L^*]+[RL^*],
\qquad
[R_T]=[R]+[RL^*].
$$

Substituting $[RL^*]=[R][L^*]/K_d^*$ and eliminating $[L^*]$ gives a quadratic
in free receptor,

$$
[R]^2+a[R]+b=0,
\qquad
a=K_d^*+[L_T^*]-[R_T],
\qquad
b=-K_d^*[R_T].
$$

Since $b\le0$, there is exactly one non-negative root:

$$
[R]=\frac{-a+\sqrt{a^2-4b}}{2}.
$$

The code evaluates this root in an algebraically equivalent form that avoids
loss of precision when $a$ is large and positive. The complex and the bound
fraction then follow as

$$
[RL^*]=[R_T]-[R],
\qquad
F_b^*=\frac{[RL^*]}{[L_T^*]}.
$$

### Direct binding with nonspecific tracer binding

Model: `dir_total`

Here, part of the tracer also binds nonspecifically, for example to the plate
or to other proteins. The nonspecifically bound amount is taken to be
proportional to free tracer:

$$
[L^*]_{\mathrm{NS}}=N^*[L^*],
\qquad N^*\ge0.
$$

The tracer balance becomes

$$
[L_T^*]=(1+N^*)[L^*]+[RL^*],
$$

while the receptor balance is unchanged, $[R_T]=[R]+[RL^*]$. Free receptor
again follows from the quadratic above, now with

$$
a=(1+N^*)K_d^*+[L_T^*]-[R_T],
\qquad
b=-(1+N^*)K_d^*[R_T].
$$

The total fraction of tracer that is bound, specifically or not, is

$$
F_{b,\mathrm{total}}^*
=\frac{[RL^*]+N^*[L^*]}{[L_T^*]}.
$$

Without receptor this fraction is $F_0=N^*/(1+N^*)$, and `ymin` is fitted to
that baseline. The model therefore maps the bound fraction measured above the
baseline to the response:

$$
\frac{F_{b,\mathrm{total}}^*-F_0}{1-F_0}
=\frac{[RL^*]}{[L_T^*]}.
$$

The factor $(1+N^*)$ shifts the curve because it changes the tracer balance.
It does not change the microscopic $K_d^*$, and total binding is not simply
$(1+N^*)$ times specific binding.

`BindingModel.species` reports the nonspecifically bound tracer as
`Ls_nonspecific`, next to the specific fraction `Fbs`.

## Complete competition: three states

Models: `comp_3st_specific`, `comp_3st_total`

In complete competition, tracer and competitor cannot bind the receptor at the
same time. Competitor is titrated into a fixed amount of receptor and tracer:

$$
R+L^*\rightleftharpoons RL^*,
\qquad
R+L\rightleftharpoons RL,
$$

$$
K_d^*=\frac{[R][L^*]}{[RL^*]},
\qquad
K_d=\frac{[R][L]}{[RL]}.
$$

The mass balances are

$$
\begin{aligned}
[L_T^*]&=[L^*]+[RL^*],\\
[L_T]&=[L]+[RL],\\
[R_T]&=[R]+[RL^*]+[RL].
\end{aligned}
$$

The two ligand balances give both complexes in terms of free receptor:

$$
[RL^*]=\frac{[L_T^*][R]}{K_d^*+[R]},
\qquad
[RL]=\frac{[L_T][R]}{K_d+[R]}.
$$

The bound fraction is therefore

$$
F_b^*=\frac{[RL^*]}{[L_T^*]}
     =\frac{[R]}{K_d^*+[R]}.
$$

It contains the tracer constant $K_d^*$, not $K_d$. The competitor acts only
by lowering $[R]$.

Substituting both complexes into the receptor balance gives a cubic in $[R]$,

$$
[R]^3+a[R]^2+b[R]+c=0,
$$

with

$$
\begin{aligned}
a&=K_d^*+K_d+[L_T^*]+[L_T]-[R_T],\\
b&=K_d^*([L_T]-[R_T])+K_d([L_T^*]-[R_T])+K_d^*K_d,\\
c&=-K_d^*K_d[R_T].
\end{aligned}
$$

Wang (1995) gives a closed-form solution of this cubic. `bindcurve` does not
use it. Instead it finds the root of the receptor balance on
$0\le[R]\le[R_T]$ numerically. The balance increases monotonically in $[R]$,
so the root is unique, and the numerical search avoids the choice between
cubic roots and the rounding errors of the closed form.

In `comp_3st_total`, competitor also binds nonspecifically,
$[L]_{\mathrm{NS}}=N[L]$, and the competitor balance becomes

$$
[L_T]=(1+N)[L]+[RL].
$$

This changes the expression for the competitor complex to
$[RL]=[L_T][R]/\big((1+N)K_d+[R]\big)$. The receptor balance is the same as
before with $K_d$ replaced by $(1+N)K_d$. The fitted $K_d$ is still the
microscopic dissociation constant of the specific complex.

## Incomplete competition: four states

Models: `comp_4st_specific`, `comp_4st_total`

In incomplete competition, tracer and competitor can bind the receptor at the
same time and form a ternary complex:

$$
\begin{aligned}
R+L^*&\rightleftharpoons RL^*,\\
R+L&\rightleftharpoons RL,\\
RL+L^*&\rightleftharpoons RLL^*,\\
RL^*+L&\rightleftharpoons RLL^*.
\end{aligned}
$$

The first three equilibria define the model parameters:

$$
K_d^*=\frac{[R][L^*]}{[RL^*]},
\qquad
K_d=\frac{[R][L]}{[RL]},
\qquad
K_{d3}=\frac{[RL][L^*]}{[RLL^*]}.
$$

The fourth constant is not free. The four reactions form a closed cycle, and at
equilibrium the constants around a cycle must be consistent, which fixes

$$
K_{d4}=\frac{[RL^*][L]}{[RLL^*]}
       =\frac{K_dK_{d3}}{K_d^*}.
$$

The mass balances are

$$
\begin{aligned}
[L_T^*]&=[L^*]+[RL^*]+[RLL^*],\\
[L_T]&=[L]+[RL]+[RLL^*],\\
[R_T]&=[R]+[RL^*]+[RL]+[RLL^*],
\end{aligned}
$$

and the complexes follow from the free concentrations:

$$
[RL^*]=\frac{[R][L^*]}{K_d^*},
\qquad
[RL]=\frac{[R][L]}{K_d},
\qquad
[RLL^*]=\frac{[R][L][L^*]}{K_dK_{d3}}.
$$

Tracer is bound in two complexes, so the bound fraction counts both:

$$
F_b^*=\frac{[RL^*]+[RLL^*]}{[L_T^*]}.
$$

Eliminating the free ligands from the mass balances gives a quintic in $[R]$,
which in general has no closed-form solution. `bindcurve` solves the system in two
nested steps instead. For a trial value of $[R]$, the two ligand balances
reduce to a quadratic with a single non-negative root, which gives $[L]$ and
$[L^*]$. The receptor balance is then solved for $[R]$ on
$0\le[R]\le[R_T]$. The balance is negative at $[R]=0$ and non-negative at
$[R]=[R_T]$, and the system has exactly one positive equilibrium, so this
interval always contains the physical root.

A few limiting cases are useful for checking results:

- Without competitor, $[L_T]=0$, the model is direct tracer binding with
  $K_d^*$.
- With a large excess of competitor, nearly all receptor is in $RL$ or
  $RLL^*$, and tracer binds $RL$ with constant $K_{d3}$:

  $$
  \lim_{[L_T]\to\infty}F_b^*
  =\frac{K_{d3}+[L_T^*]+[R_T]
  -\sqrt{(K_{d3}+[L_T^*]+[R_T])^2-4[L_T^*][R_T]}}
  {2[L_T^*]}.
  $$

  This is nonzero for any finite $K_{d3}$, so the curve levels off above zero.
- If $K_{d3}=K_d^*$, competitor does not affect tracer binding at all, and
  $F_b^*$ is the same at every competitor concentration.
- As $K_{d3}\to\infty$, the ternary complex disappears and the model becomes
  the three-state model.

When $K_{d3}>K_d^*$, competitor weakens tracer binding and $F_b^*$ falls with
competitor concentration, but not to zero. When $K_{d3}<K_d^*$, competitor
strengthens tracer binding and $F_b^*$ rises.

In `comp_4st_total`, competitor also binds nonspecifically, and the competitor
balance becomes

$$
[L_T]=(1+N)[L]+[RL]+[RLL^*].
$$

Written in terms of the apparent free competitor $[L]_{\mathrm{app}}=(1+N)[L]$,
all three mass balances take the form of `comp_4st_specific` with $K_d$
replaced by $(1+N)K_d$. For example,

$$
[RL]=\frac{[R][L]}{K_d}=\frac{[R][L]_{\mathrm{app}}}{(1+N)K_d}.
$$

Both substitutions are needed; replacing $K_d$ alone would give the wrong
$[RL]$ and $[RLL^*]$. `bindcurve` solves this equivalent system and converts
$[L]_{\mathrm{app}}$ back to $[L]$, so the fitted $K_d$ is still the
microscopic constant.

## References

- Wang, Z.-X. (1995), [An exact mathematical expression for describing
  competitive binding of two different ligands to a protein
  molecule](https://doi.org/10.1016/0014-5793(95)00062-E).
- Roehrl, M. H. A., Wang, J. Y. & Wagner, G. (2004), [A general framework for
  development and data analysis of competitive high-throughput screens for
  small-molecule inhibitors of protein-protein interactions by fluorescence
  polarization](https://doi.org/10.1021/bi048233g).
