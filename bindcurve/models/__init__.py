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
    """Return a built-in model by name.

    Parameters
    ----------
    name
        One of ``"ic50"``, ``"dir_simple"``, ``"dir_specific"``,
        ``"dir_total"``, ``"comp_3st_specific"``, ``"comp_3st_total"``,
        ``"comp_4st_specific"`` or ``"comp_4st_total"``. The models and their
        required assay constants are listed in :doc:`/user-guide/models`.

    Raises
    ------
    KeyError
        If there is no model of that name.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}; choose from {sorted(MODELS)}."
        ) from None


__all__ = ["MODELS", "BindingModel", "Model", "Parameter", "get_model"]
