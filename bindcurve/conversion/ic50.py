from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

ConversionModel = Literal["cheng_prusoff", "cheng_prusoff_corrected", "coleska"]


@dataclass(frozen=True)
class IC50ConversionResult:
    """Result of one IC50-to-Kd conversion."""

    compound_id: str | None
    model: str
    IC50: float
    Kd: float
    lower_IC50: float | None = None
    upper_IC50: float | None = None
    lower_Kd: float | None = None
    upper_Kd: float | None = None


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _require_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def cheng_prusoff(*, IC50: float, LsT: float, Kds: float) -> float:
    """Convert IC50 to Kd with the Cheng-Prusoff approximation."""
    IC50 = _require_positive("IC50", IC50)
    LsT = _require_positive("LsT", LsT)
    Kds = _require_positive("Kds", Kds)
    return IC50 / (1.0 + LsT / Kds)


def cheng_prusoff_corrected(
    *,
    IC50: float,
    LsT: float,
    Kds: float,
    y0: float,
) -> float:
    """Convert IC50 to Kd with the exact Munson-Rodbard correction.

    Exactness is conditional on its one-site competitive-equilibrium model.
    ``y0`` is the bound-to-free tracer ratio before competitor is added.
    """
    IC50 = _require_positive("IC50", IC50)
    LsT = _require_positive("LsT", LsT)
    Kds = _require_positive("Kds", Kds)
    y0 = _require_nonnegative("y0", y0)
    denominator = 1.0 + (LsT * (y0 + 2.0) / (2.0 * Kds * (y0 + 1.0)) + y0)
    # Munson & Rodbard, J Receptor Res. 1989-90;9(6):511, erratum
    # (doi:10.3109/10799898909066075): the printed plus sign is a minus sign.
    Kd = IC50 / denominator - Kds * (y0 / (y0 + 2.0))
    if not np.isfinite(Kd) or Kd <= 0.0:
        raise ValueError(
            "Converted Kd is non-positive or non-finite. Check whether IC50 "
            "is physically compatible with LsT, Kds, and y0."
        )
    return float(Kd)


def coleska(*, IC50: float, RT: float, LsT: float, Kds: float) -> float:
    """Convert IC50 to Kd with the Coleska finite-concentration correction."""
    IC50 = _require_positive("IC50", IC50)
    RT = _require_positive("RT", RT)
    LsT = _require_positive("LsT", LsT)
    Kds = _require_positive("Kds", Kds)

    a = LsT + Kds - RT
    b = -Kds * RT
    square_root = np.hypot(a, 2.0 * np.sqrt(-b))
    if a >= 0.0:
        R0 = -2.0 * b / (a + square_root)
    else:
        R0 = (-a + square_root) / 2.0

    Ls0 = LsT / (1.0 + R0 / Kds)
    RLs0 = RT / (1.0 + Kds / Ls0)
    RLs50 = RLs0 / 2.0
    Ls50 = LsT - RLs50
    # Receptor mass balance from Appendix A.4.2 of Nikolovska-Coleska et al.,
    # Anal Biochem. 2004;332:261-273. doi:10.1016/j.ab.2004.05.055.
    RL50 = RT - Kds * (RLs50 / Ls50) - RLs50
    L50 = IC50 - RL50
    Kd = L50 / ((Ls50 / Kds) + (R0 / Kds) + 1.0)

    if not np.isfinite(Kd) or Kd <= 0.0:
        raise ValueError(
            "Converted Kd is non-positive or non-finite. Check whether IC50 "
            "is physically compatible with RT, LsT, and Kds."
        )
    return float(Kd)


def _convert_scalar(
    *,
    IC50: float,
    model: ConversionModel,
    RT: float | None,
    LsT: float | None,
    Kds: float | None,
    y0: float | None,
) -> float:
    if model == "cheng_prusoff":
        if LsT is None or Kds is None:
            raise ValueError("cheng_prusoff requires LsT and Kds.")
        return cheng_prusoff(IC50=IC50, LsT=LsT, Kds=Kds)

    if model == "cheng_prusoff_corrected":
        if LsT is None or Kds is None or y0 is None:
            raise ValueError("cheng_prusoff_corrected requires LsT, Kds, and y0.")
        return cheng_prusoff_corrected(IC50=IC50, LsT=LsT, Kds=Kds, y0=y0)

    if model == "coleska":
        if RT is None or LsT is None or Kds is None:
            raise ValueError("coleska requires RT, LsT, and Kds.")
        return coleska(IC50=IC50, RT=RT, LsT=LsT, Kds=Kds)

    raise ValueError(f"Unknown conversion model: {model!r}")


def convert_ic50_to_kd(
    data: pd.DataFrame | None = None,
    *,
    model: ConversionModel,
    IC50: float | None = None,
    RT: float | None = None,
    LsT: float | None = None,
    Kds: float | None = None,
    y0: float | None = None,
    compound_col: str = "compound_id",
    ic50_col: str = "IC50",
    lower_col: str | None = None,
    upper_col: str | None = None,
) -> IC50ConversionResult | pd.DataFrame:
    """Convert scalar or DataFrame IC50 values to Kd.

    When ``data`` is provided, the return value is a DataFrame with one row per
    input row. Otherwise, ``IC50`` must be provided and a single
    ``IC50ConversionResult`` is returned.
    """
    if model not in {"cheng_prusoff", "cheng_prusoff_corrected", "coleska"}:
        raise ValueError(f"Unknown conversion model: {model!r}")

    if data is None:
        if IC50 is None:
            raise ValueError("Either data or IC50 must be provided.")
        converted = _convert_scalar(
            IC50=IC50,
            model=model,
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            y0=y0,
        )
        return IC50ConversionResult(
            compound_id=None,
            model=model,
            IC50=float(IC50),
            Kd=converted,
        )

    required_columns = [compound_col, ic50_col]
    if lower_col is not None:
        required_columns.append(lower_col)
    if upper_col is not None:
        required_columns.append(upper_col)
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing_columns}."
        )
    if data[compound_col].isna().any():
        raise ValueError(f"{compound_col!r} contains missing compound identifiers.")

    rows = []
    for _, row in data.iterrows():
        ic50_value = float(row[ic50_col])
        kd_value = _convert_scalar(
            IC50=ic50_value,
            model=model,
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            y0=y0,
        )

        lower_ic50 = None if lower_col is None else float(row[lower_col])
        upper_ic50 = None if upper_col is None else float(row[upper_col])
        if lower_ic50 is not None and lower_ic50 > ic50_value:
            raise ValueError("lower IC50 must not exceed the central IC50 value.")
        if upper_ic50 is not None and upper_ic50 < ic50_value:
            raise ValueError("upper IC50 must not be below the central IC50 value.")
        lower_kd = None
        upper_kd = None
        if lower_ic50 is not None:
            lower_kd = _convert_scalar(
                IC50=lower_ic50,
                model=model,
                RT=RT,
                LsT=LsT,
                Kds=Kds,
                y0=y0,
            )
        if upper_ic50 is not None:
            upper_kd = _convert_scalar(
                IC50=upper_ic50,
                model=model,
                RT=RT,
                LsT=LsT,
                Kds=Kds,
                y0=y0,
            )

        rows.append(
            asdict(
                IC50ConversionResult(
                compound_id=str(row[compound_col]),
                model=model,
                IC50=ic50_value,
                Kd=kd_value,
                lower_IC50=lower_ic50,
                upper_IC50=upper_ic50,
                lower_Kd=lower_kd,
                upper_Kd=upper_kd,
                )
            )
        )

    return pd.DataFrame(rows)
