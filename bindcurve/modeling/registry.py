from __future__ import annotations

from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.binding import (
    CompetitiveThreeStateSpecificKdModel,
    CompetitiveThreeStateTotalKdModel,
    DirectSimpleKdModel,
    DirectSpecificKdModel,
    DirectTotalKdModel,
)
from bindcurve.modeling.logistic import EC50Model, IC50Model, LogIC50Model

_MODELS: dict[str, BaseDoseResponseModel] = {
    IC50Model.name: IC50Model(),
    EC50Model.name: EC50Model(),
    LogIC50Model.name: LogIC50Model(),
    DirectSimpleKdModel.name: DirectSimpleKdModel(),
    DirectSpecificKdModel.name: DirectSpecificKdModel(),
    DirectTotalKdModel.name: DirectTotalKdModel(),
    CompetitiveThreeStateSpecificKdModel.name: CompetitiveThreeStateSpecificKdModel(),
    CompetitiveThreeStateTotalKdModel.name: CompetitiveThreeStateTotalKdModel(),
}


def get_model(name: str) -> BaseDoseResponseModel:
    """Return a registered model by name."""
    normalized = name.lower()
    try:
        return _MODELS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS))
        raise KeyError(f"Unknown model {name!r}. Available models: {available}") from exc
