from __future__ import annotations

from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.logistic import IC50Model

_MODELS: dict[str, BaseDoseResponseModel] = {
    IC50Model.name: IC50Model(),
}


def get_model(name: str) -> BaseDoseResponseModel:
    """Return a registered model by name."""
    normalized = name.lower()
    try:
        return _MODELS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS))
        raise KeyError(f"Unknown model {name!r}. Available models: {available}") from exc
