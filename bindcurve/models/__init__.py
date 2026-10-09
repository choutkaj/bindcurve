"""Dose-response and equilibrium-binding models."""

from __future__ import annotations

from bindcurve.models.base import BindingModel, Model, Parameter
from bindcurve.models.direct import DirectModel, DirectSimpleModel
from bindcurve.models.four_state import FourStateModel
from bindcurve.models.logistic import IC50Model
from bindcurve.models.three_state import ThreeStateModel

MODELS: dict[str, Model] = {
    model.name: model
    for model in (
        IC50Model(),
        DirectSimpleModel(),
        DirectModel(nonspecific=False),
        DirectModel(nonspecific=True),
        ThreeStateModel(nonspecific=False),
        ThreeStateModel(nonspecific=True),
        FourStateModel(nonspecific=False),
        FourStateModel(nonspecific=True),
    )
}


def get_model(name: str) -> Model:
    """Return a built-in model by name, e.g. ``"ic50"`` or ``"comp_4st_specific"``."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}; choose from {sorted(MODELS)}."
        ) from None


__all__ = ["MODELS", "BindingModel", "Model", "Parameter", "get_model"]
