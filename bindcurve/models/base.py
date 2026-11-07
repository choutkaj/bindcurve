"""Model and parameter definitions shared by all dose-response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """One model parameter.

    Parameters
    ----------
    name
        Keyword used in `Model.evaluate` and in the ``fixed`` and ``bounds``
        arguments of `fit`.
    concentration
        Whether this is a positive, concentration-like parameter. Such
        parameters are optimized on a log10 coordinate and summarized across
        experiments on the log10 scale.
    fixed
        Whether this is an assay constant that must be given in
        ``fit(..., fixed=...)``.
    min, max
        Physical limits of the parameter value.
    """

    name: str
    concentration: bool = False
    fixed: bool = False
    min: float = -np.inf
    max: float = np.inf


PLATEAUS = (Parameter("ymin"), Parameter("ymax"))


class Model(ABC):
    """Dose-response model mapping a fraction in [0, 1] onto two plateaus.

    The response is ``ymin + (ymax - ymin) * fraction(x)``.

    Attributes
    ----------
    name
        Name of the model, as used by `get_model`.
    parameters
        All parameters, including ``ymin`` and ``ymax``.

    Notes
    -----
    A custom model sets ``name`` and ``parameters`` and implements
    ``_fraction(self, x, **params)``, which receives the concentrations as an
    array and every parameter as a keyword argument, already validated
    against its `Parameter` limits, and returns the fraction at each
    concentration. Fitted parameters other than the plateaus and
    concentrations also need a starting value from an extended `guess`. The
    model can then be passed to `fit` in place of a name.
    """

    name: str
    parameters: tuple[Parameter, ...]

    def evaluate(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the response at concentrations ``x``.

        Parameters
        ----------
        x
            Concentrations on the linear (not log) scale, ``x >= 0``.
        **params
            A value for every parameter of the model, by name.

        Raises
        ------
        TypeError
            If parameters are missing or unknown.
        ValueError
            If a concentration or parameter value is invalid.
        """
        x, params = self._checked(x, params)
        fraction = self._fraction(x, **params)
        return params["ymin"] + (params["ymax"] - params["ymin"]) * fraction

    def guess(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Initial values for fitting replicate means ``y`` at concentrations ``x``.

        The plateaus come from the response range and every fitted
        concentration parameter from the concentration closest to the half
        response. Subclasses with other fitted parameters extend the result.

        Parameters
        ----------
        x
            Concentrations of one experiment.
        y
            Replicate mean response at each concentration.
        """
        half = np.argmin(np.abs(y - (np.min(y) + np.max(y)) / 2.0))
        guesses = {"ymin": float(np.min(y)), "ymax": float(np.max(y))}
        for parameter in self.parameters:
            if parameter.concentration and not parameter.fixed:
                guesses[parameter.name] = float(x[half])
        return guesses

    def parameter(self, name: str) -> Parameter:
        """Return the parameter called ``name``.

        Parameters
        ----------
        name
            Parameter name, e.g. ``"Kd"``.

        Raises
        ------
        KeyError
            If the model has no such parameter.
        """
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"Model {self.name!r} has no parameter {name!r}.")

    @abstractmethod
    def _fraction(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Return the modeled fraction for validated inputs."""

    def _checked(
        self, x: np.ndarray, params: dict[str, float]
    ) -> tuple[np.ndarray, dict[str, float]]:
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x < 0.0):
            raise ValueError("Concentrations must be finite and non-negative.")
        expected = [parameter.name for parameter in self.parameters]
        if set(params) != set(expected):
            raise TypeError(
                f"Model {self.name!r} takes {expected}; got {sorted(params)}."
            )
        checked = {}
        for parameter in self.parameters:
            value = float(params[parameter.name])
            if not np.isfinite(value) or not parameter.min <= value <= parameter.max:
                raise ValueError(
                    f"{parameter.name} = {value} is outside "
                    f"[{parameter.min}, {parameter.max}]."
                )
            if parameter.concentration and value <= 0.0:
                raise ValueError(f"{parameter.name} must be positive.")
            checked[parameter.name] = value
        return x, checked


class BindingModel(Model):
    """Equilibrium-binding model whose fraction is the tracer-bound fraction.

    Notes
    -----
    A custom binding model implements ``_species(self, x, **params)`` in place
    of ``_fraction``. It returns a dict of arrays that includes the bound
    fraction ``"Fbs"``.
    """

    def species(self, x: np.ndarray, **params: float) -> dict[str, np.ndarray]:
        """Return the equilibrium concentrations of all species.

        Parameters
        ----------
        x
            Concentrations of the titrated component, as in `Model.evaluate`.
        **params
            A value for every parameter of the model, by name.

        Returns
        -------
        dict[str, numpy.ndarray]
            Arrays shaped like ``x``, in the unit of the concentrations:

            - ``R``: free receptor
            - ``Ls``: free tracer
            - ``RLs``: tracer-receptor complex
            - ``L``: free competitor (competitive models)
            - ``RL``: competitor-receptor complex (competitive models)
            - ``RLLs``: ternary complex (four-state models)
            - ``Ls_nonspecific``: nonspecifically bound tracer (``dir_total``)
            - ``L_nonspecific``: nonspecifically bound competitor
              (``comp_*_total``)
            - ``Fbs``: fraction of tracer bound to receptor (dimensionless)

            ``dir_simple`` returns only ``R`` and ``Fbs``.
        """
        x, params = self._checked(x, params)
        return self._species(x, **params)

    def _fraction(self, x: np.ndarray, **params: float) -> np.ndarray:
        return self._species(x, **params)["Fbs"]

    @abstractmethod
    def _species(self, x: np.ndarray, **params: float) -> dict[str, np.ndarray]:
        """Return species for validated inputs."""
