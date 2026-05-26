from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes

from bindcurve.datasets import DoseResponseData
from bindcurve.plotting.common import (
    _evaluate_fit,
    _get_axes,
    _matching_fits,
    _resolve_compound_ids,
)
from bindcurve.results import FitResults

AsymptoteName = Literal["ymin", "ymax"]
CurvePointSpec = float | tuple[float, str] | dict[str, object]


@dataclass(frozen=True)
class CurvePoint:
    """Point to annotate on a fitted curve."""

    x: float
    label: str | None = None


def _coerce_curve_points(points: Iterable[CurvePointSpec]) -> list[CurvePoint]:
    coerced = []
    for point in points:
        if isinstance(point, dict):
            if "x" not in point:
                raise ValueError("Curve point dictionaries must contain an 'x' key.")
            coerced.append(
                CurvePoint(
                    x=float(point["x"]),
                    label=None if point.get("label") is None else str(point["label"]),
                )
            )
        elif isinstance(point, tuple):
            if len(point) != 2:
                raise ValueError("Curve point tuples must be (x, label).")
            coerced.append(CurvePoint(x=float(point[0]), label=str(point[1])))
        else:
            coerced.append(CurvePoint(x=float(point), label=None))
    return coerced


def plot_asymptotes(
    data: DoseResponseData,
    results: FitResults,
    *,
    compound_id: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    parameters: Sequence[AsymptoteName] = ("ymin", "ymax"),
    label: bool = True,
    **line_kwargs,
) -> Axes:
    """Plot model asymptotes as horizontal lines on an existing axes."""
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)
    fits = _matching_fits(
        results,
        compound_ids=resolved_compound_ids,
        experiments=experiments,
    )

    default_kwargs = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7}
    default_kwargs.update(line_kwargs)

    n_drawn = 0
    for fit in fits:
        for parameter in parameters:
            if parameter not in fit.parameters:
                continue
            asymptote_label = None
            if label:
                experiment = fit.experiment_id or fit.model_name
                if len(resolved_compound_ids) > 1:
                    asymptote_label = f"{fit.compound_id} {experiment} {parameter}"
                else:
                    asymptote_label = f"{experiment} {parameter}"
            ax.axhline(
                fit.parameters[parameter].value,
                label=asymptote_label,
                **default_kwargs,
            )
            n_drawn += 1

    if n_drawn == 0:
        raise ValueError("No requested asymptote parameters were available to plot.")

    return ax


def plot_curve_points(
    data: DoseResponseData,
    results: FitResults,
    *,
    points: Iterable[CurvePointSpec],
    compound_id: str | Iterable[str] | None = None,
    ax: Axes | None = None,
    experiments: Iterable[str] | None = None,
    annotate: bool = True,
    annotation_offset: tuple[float, float] = (6.0, 6.0),
    point_kwargs: dict | None = None,
    annotation_kwargs: dict | None = None,
) -> Axes:
    """Plot arbitrary labeled points evaluated on fitted curves."""
    ax = _get_axes(ax)
    resolved_compound_ids = _resolve_compound_ids(data, compound_id)
    fits = _matching_fits(
        results,
        compound_ids=resolved_compound_ids,
        experiments=experiments,
    )
    curve_points = _coerce_curve_points(points)

    default_point_kwargs = {"marker": "o", "zorder": 5}
    default_point_kwargs.update(point_kwargs or {})
    default_annotation_kwargs = {"textcoords": "offset points"}
    default_annotation_kwargs.update(annotation_kwargs or {})

    for fit in fits:
        for point in curve_points:
            y_value = float(np.asarray(_evaluate_fit(fit, point.x)))
            ax.scatter([point.x], [y_value], **default_point_kwargs)

            if not annotate or point.label is None:
                continue

            label = point.label
            if len(fits) > 1 or len(resolved_compound_ids) > 1:
                experiment = fit.experiment_id or fit.model_name
                if len(resolved_compound_ids) > 1:
                    label = f"{label} ({fit.compound_id} {experiment})"
                else:
                    label = f"{label} ({experiment})"

            ax.annotate(
                label,
                xy=(point.x, y_value),
                xytext=annotation_offset,
                **default_annotation_kwargs,
            )

    return ax
