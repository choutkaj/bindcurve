"""Derive the four-state competitive-binding quintic used by bindcurve.

The Roehrl, Wang, and Wagner paper gives the incomplete competitive-binding
model as a quintic in FSB, the fraction of bound labeled ligand.  The legacy
bindcurve implementation instead used a quintic in the transformed coordinate

    r_equiv = Kds * FSB / (1 - FSB)

This script shows the relationship between the paper equation and the
implementation coefficients.

Important terminology:

- ``r_equiv`` is *not* generally the physical free receptor concentration in the
  four-state model. It is an equivalent receptor coordinate that maps back to
  FSB through ``FSB = r_equiv / (Kds + r_equiv)``.
- In the three-state complete-competition model, this coordinate coincides with
  free receptor. In the four-state incomplete model, it should be interpreted as
  a transformed bound-fraction coordinate.
- Nonspecific competitor binding is handled by substituting
  ``Kd -> (1 + N) * Kd`` into the specific four-state coefficients, matching
  equation 30 of the paper.

Run this file directly to print the derived coefficients.
"""

from __future__ import annotations

import sympy as sp


# Paper notation, mapped to bindcurve names:
#   KD1 -> Kds: tracer/receptor dissociation constant
#   KD2 -> Kd:  competitor/receptor dissociation constant
#   KD3 -> Kd3: tracer/(receptor-competitor) dissociation constant
#   LT  -> competitor total concentration
#   LsT -> tracer total concentration
#   RT  -> receptor total concentration
#   FSB -> bound tracer fraction
Kds, Kd, Kd3, LT, LsT, RT, FSB, r_equiv, N = sp.symbols(
    "Kds Kd Kd3 LT LsT RT FSB r_equiv N",
)


# Equation 27 of Roehrl et al. written as LT = numerator / denominator.
j = LsT * FSB**2 - (Kds + LsT + RT) * FSB + RT
k = LsT * FSB**2 - (Kd3 + LsT + RT) * FSB + RT
l = LsT * FSB - Kd3 - LsT

paper_eq27_residual = sp.expand(
    LT * k * (1 - FSB) * (Kds - Kd3) * Kds
    - j * ((k * l - (1 - FSB) * Kd * Kd3) * Kds + (1 - FSB) * Kd * Kd3**2)
)


# Substitute FSB = r_equiv / (Kds + r_equiv).  This is equivalent to
# r_equiv = Kds * FSB / (1 - FSB).
substituted = sp.together(
    paper_eq27_residual.subs(FSB, r_equiv / (Kds + r_equiv))
)
numerator = sp.factor(substituted.as_numer_denom()[0])

# Remove denominator-clearing factors that do not change the roots.
polynomial = sp.factor(numerator / (Kds**2 * (Kds + r_equiv) ** 10))
poly = sp.Poly(polynomial, r_equiv)


# The implementation stores the negative, Kds/Kd3-scaled form below because it
# matches the historical legacy coefficient convention.  Multiplication by a
# nonzero scalar does not change polynomial roots.
implementation_form = sp.factor(-Kds**2 * Kd3**2 * polynomial / poly.LC())
implementation_poly = sp.Poly(implementation_form, r_equiv)
implementation_coefficients = [sp.factor(c) for c in implementation_poly.all_coeffs()]


# Nonspecific competitor binding: paper equation 30 states that equation 27 is
# identical except that KD2 is replaced everywhere by (1 + N) KD2.
implementation_coefficients_total = [
    sp.factor(c.subs(Kd, (1 + N) * Kd)) for c in implementation_coefficients
]


def main() -> None:
    print("Specific four-state coefficients in r_equiv:")
    for name, coefficient in zip("abcdef", implementation_coefficients, strict=True):
        print(f"{name} = {coefficient}")

    print("\nTotal/nonspecific four-state coefficients use Kd_eff = (1 + N) * Kd:")
    for name, coefficient in zip("abcdef", implementation_coefficients_total, strict=True):
        print(f"{name} = {coefficient}")


if __name__ == "__main__":
    main()
