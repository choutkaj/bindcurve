from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from bindcurve.datasets import DoseResponseData
    from bindcurve.results import FitResults


def plot_curves(
    data: DoseResponseData,
    results: FitResults,
    *,
    ax: Axes | None = None,
):
    """Plot dose-response data and fitted curves.

    This is a placeholder for the future plotting API. It exists so that the
    package namespace is ready for plotting, but fitting and result semantics can
    stabilize before plotting behavior is designed in detail.
    """
    raise NotImplementedError(
        "plot_curves is not implemented yet. Plotting will be added after the "
        "core fitting API stabilizes."
    )
