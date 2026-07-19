from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData


def midpoint_guess(
    compound: CompoundData,
    *,
    concentration_parameter: str,
) -> dict[str, float]:
    """Return asymptote and midpoint-concentration guesses for one experiment."""
    table = compound.aggregate_replicates()
    concentration = table["concentration"].to_numpy(dtype=float)
    response = table["response"].to_numpy(dtype=float)
    ymin = float(np.min(response))
    ymax = float(np.max(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.argmin(np.abs(response - midpoint)))
    return {
        "ymin": ymin,
        "ymax": ymax,
        concentration_parameter: float(concentration[midpoint_index]),
    }
