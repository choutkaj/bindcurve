"""Plotting entry points for dose-response results.

The plotting API is intentionally minimal at this stage. Full plotting support will
be implemented after the data/model/fitting/result interfaces are stable.
"""

from bindcurve.plotting.curves import plot_curves

__all__ = ["plot_curves"]
