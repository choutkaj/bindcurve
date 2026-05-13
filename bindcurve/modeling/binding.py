from __future__ import annotations

import numpy as np

from bindcurve.datasets import CompoundData
from bindcurve.modeling.base import BaseDoseResponseModel
from bindcurve.modeling.parameters import ConcentrationParameterSpec, ParameterSpec


def _aggregate_for_guess(compound: CompoundData) -> tuple[np.ndarray, np.ndarray]:
    table = compound.aggregate_replicates()
    concentration = table["concentration"].to_numpy(dtype=float)
    response = table["response"].to_numpy(dtype=float)
    return concentration, response


def _basic_guess(compound: CompoundData) -> dict[str, float]:
    concentration, response = _aggregate_for_guess(compound)
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
    kds_guess = float(concentration[midpoint_index])
    return {"ymin": ymin, "ymax": ymax, "Kds": kds_guess}


def _competition_guess(compound: CompoundData) -> dict[str, float]:
    concentration, response = _aggregate_for_guess(compound)
    ymin = float(np.nanmin(response))
    ymax = float(np.nanmax(response))
    midpoint = ymin + 0.5 * (ymax - ymin)
    midpoint_index = int(np.nanargmin(np.abs(response - midpoint)))
    kd_guess = float(concentration[midpoint_index])
    return {"ymin": ymin, "ymax": ymax, "Kd": kd_guess}


def _competitive_three_state_receptor_free(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
    nonspecific_factor: float = 1.0,
) -> np.ndarray:
    ligand_total = np.asarray(ligand_total, dtype=float)
    scaled_kd = nonspecific_factor * Kd

    a = Kds + scaled_kd + LsT + ligand_total - RT
    b = Kds * (ligand_total - RT) + scaled_kd * (LsT - RT) + Kds * scaled_kd
    c = -Kds * scaled_kd * RT

    discriminant = np.maximum(a**2 - 3.0 * b, 0.0)
    denominator = 2.0 * np.sqrt(discriminant**3)
    numerator = -2.0 * a**3 + 9.0 * a * b - 27.0 * c
    argument = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )
    theta = np.arccos(np.clip(argument, -1.0, 1.0))
    return -(a / 3.0) + (2.0 / 3.0) * np.sqrt(discriminant) * np.cos(theta / 3.0)


def _direct_specific_receptor_free(
    receptor_total: np.ndarray,
    *,
    LsT: float,
    Kds: float,
) -> np.ndarray:
    receptor_total = np.asarray(receptor_total, dtype=float)
    a = Kds + LsT - receptor_total
    b = -Kds * receptor_total
    return (-a + np.sqrt(np.maximum(a**2 - 4.0 * b, 0.0))) / 2.0


def _direct_total_receptor_free(
    receptor_total: np.ndarray,
    *,
    LsT: float,
    Ns: float,
    Kds: float,
) -> np.ndarray:
    receptor_total = np.asarray(receptor_total, dtype=float)
    a = (1.0 + Ns) * Kds + LsT - receptor_total
    b = -Kds * receptor_total * (1.0 + Ns)
    return (-a + np.sqrt(np.maximum(a**2 - 4.0 * b, 0.0))) / 2.0


def _direct_specific_component_arrays(
    receptor_total: np.ndarray,
    *,
    LsT: float,
    Kds: float,
) -> dict[str, np.ndarray]:
    receptor_total = np.asarray(receptor_total, dtype=float)
    receptor_free = _direct_specific_receptor_free(
        receptor_total,
        LsT=LsT,
        Kds=Kds,
    )
    tracer_bound = np.clip(receptor_total - receptor_free, 0.0, None)
    tracer_free = np.clip(LsT - tracer_bound, 0.0, None)
    fraction_bound = np.divide(
        tracer_bound,
        LsT,
        out=np.zeros_like(tracer_bound, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "R_total": receptor_total,
        "R_free": receptor_free,
        "Lstar_total": np.full_like(receptor_total, LsT, dtype=float),
        "Lstar_free": tracer_free,
        "RLstar": tracer_bound,
        "fraction_bound": fraction_bound,
    }


def _direct_total_component_arrays(
    receptor_total: np.ndarray,
    *,
    LsT: float,
    Ns: float,
    Kds: float,
) -> dict[str, np.ndarray]:
    receptor_total = np.asarray(receptor_total, dtype=float)
    receptor_free = _direct_total_receptor_free(
        receptor_total,
        LsT=LsT,
        Ns=Ns,
        Kds=Kds,
    )
    tracer_bound_specific = np.clip(receptor_total - receptor_free, 0.0, None)
    tracer_bound_nonspecific = Ns * tracer_bound_specific
    tracer_bound_total = tracer_bound_specific + tracer_bound_nonspecific
    tracer_free = np.clip(LsT - tracer_bound_total, 0.0, None)
    fraction_bound = np.divide(
        receptor_free,
        Kds + receptor_free,
        out=np.zeros_like(receptor_free, dtype=float),
        where=(Kds + receptor_free) != 0.0,
    )
    return {
        "R_total": receptor_total,
        "R_free": receptor_free,
        "Lstar_total": np.full_like(receptor_total, LsT, dtype=float),
        "Lstar_free": tracer_free,
        "RLstar": tracer_bound_specific,
        "Lstar_nonspecific_bound": tracer_bound_nonspecific,
        "Lstar_bound_total": tracer_bound_total,
        "fraction_bound": fraction_bound,
    }


def _competitive_three_state_specific_component_arrays(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    Kd: float,
) -> dict[str, np.ndarray]:
    ligand_total = np.asarray(ligand_total, dtype=float)
    receptor_free = _competitive_three_state_receptor_free(
        ligand_total,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
    )
    tracer_bound = np.divide(
        LsT * receptor_free,
        Kds + receptor_free,
        out=np.zeros_like(receptor_free, dtype=float),
        where=(Kds + receptor_free) != 0.0,
    )
    tracer_free = np.clip(LsT - tracer_bound, 0.0, None)
    competitor_bound = np.divide(
        ligand_total * receptor_free,
        Kd + receptor_free,
        out=np.zeros_like(receptor_free, dtype=float),
        where=(Kd + receptor_free) != 0.0,
    )
    competitor_free = np.clip(ligand_total - competitor_bound, 0.0, None)
    fraction_tracer_bound = np.divide(
        tracer_bound,
        LsT,
        out=np.zeros_like(tracer_bound, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "L_total": ligand_total,
        "R_total": np.full_like(ligand_total, RT, dtype=float),
        "R_free": receptor_free,
        "Lstar_total": np.full_like(ligand_total, LsT, dtype=float),
        "Lstar_free": tracer_free,
        "L_free": competitor_free,
        "RLstar": tracer_bound,
        "RL": competitor_bound,
        "fraction_tracer_bound": fraction_tracer_bound,
    }


def _competitive_three_state_total_component_arrays(
    ligand_total: np.ndarray,
    *,
    RT: float,
    LsT: float,
    Kds: float,
    N: float,
    Kd: float,
) -> dict[str, np.ndarray]:
    ligand_total = np.asarray(ligand_total, dtype=float)
    effective_factor = 1.0 + N
    receptor_free = _competitive_three_state_receptor_free(
        ligand_total,
        RT=RT,
        LsT=LsT,
        Kds=Kds,
        Kd=Kd,
        nonspecific_factor=effective_factor,
    )
    tracer_bound = np.divide(
        LsT * receptor_free,
        Kds + receptor_free,
        out=np.zeros_like(receptor_free, dtype=float),
        where=(Kds + receptor_free) != 0.0,
    )
    tracer_free = np.clip(LsT - tracer_bound, 0.0, None)
    competitor_bound_specific = np.divide(
        ligand_total * receptor_free,
        effective_factor * Kd + receptor_free,
        out=np.zeros_like(receptor_free, dtype=float),
        where=(effective_factor * Kd + receptor_free) != 0.0,
    )
    competitor_bound_nonspecific = N * competitor_bound_specific
    competitor_bound_total = (
        competitor_bound_specific + competitor_bound_nonspecific
    )
    competitor_free = np.clip(ligand_total - competitor_bound_total, 0.0, None)
    fraction_tracer_bound = np.divide(
        tracer_bound,
        LsT,
        out=np.zeros_like(tracer_bound, dtype=float),
        where=LsT != 0.0,
    )
    return {
        "L_total": ligand_total,
        "R_total": np.full_like(ligand_total, RT, dtype=float),
        "R_free": receptor_free,
        "Lstar_total": np.full_like(ligand_total, LsT, dtype=float),
        "Lstar_free": tracer_free,
        "L_free": competitor_free,
        "RLstar": tracer_bound,
        "RL": competitor_bound_specific,
        "L_nonspecific_bound": competitor_bound_nonspecific,
        "L_bound_total": competitor_bound_total,
        "fraction_tracer_bound": fraction_tracer_bound,
    }


class DirectSimpleKdModel(BaseDoseResponseModel):
    """Simple direct-binding saturation model.

    The model assumes the x-axis is free receptor concentration and evaluates::

        fraction_bound = R / (Kds + R)
        y = ymin + (ymax - ymin) * fraction_bound
    """

    name = "dir_simple"
    concentration_parameters = frozenset({"Kds"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("Kds", min=0.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        Kds: float,
    ) -> np.ndarray:
        receptor = np.asarray(x, dtype=float)
        fraction_bound = receptor / (Kds + receptor)
        return ymin + (ymax - ymin) * fraction_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        receptor = np.asarray(concentration, dtype=float)
        kds = float(params["Kds"])
        fraction_bound = np.divide(
            receptor,
            kds + receptor,
            out=np.zeros_like(receptor, dtype=float),
            where=(kds + receptor) != 0.0,
        )
        return {
            "R_free": receptor,
            "fraction_bound": fraction_bound,
        }

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class DirectSpecificKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion for specific binding."""

    name = "dir_specific"
    required_fixed_parameters = frozenset({"LsT"})
    concentration_parameters = frozenset({"Kds", "LsT"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("LsT", min=0.0, vary=False),
        ParameterSpec("Kds", min=0.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        LsT: float,
        Kds: float,
    ) -> np.ndarray:
        receptor_total = np.asarray(x, dtype=float)
        components = _direct_specific_component_arrays(
            receptor_total,
            LsT=LsT,
            Kds=Kds,
        )
        fraction_bound = components["fraction_bound"]
        return ymin + (ymax - ymin) * fraction_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _direct_specific_component_arrays(
            concentration,
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class DirectTotalKdModel(BaseDoseResponseModel):
    """Direct-binding model with ligand depletion and nonspecific binding."""

    name = "dir_total"
    required_fixed_parameters = frozenset({"LsT", "Ns"})
    concentration_parameters = frozenset({"Kds", "LsT"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("LsT", min=0.0, vary=False),
        ParameterSpec("Ns", min=0.0, vary=False),
        ParameterSpec("Kds", min=0.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        LsT: float,
        Ns: float,
        Kds: float,
    ) -> np.ndarray:
        receptor_total = np.asarray(x, dtype=float)
        components = _direct_total_component_arrays(
            receptor_total,
            LsT=LsT,
            Ns=Ns,
            Kds=Kds,
        )
        fraction_bound = components["fraction_bound"]
        return ymin + (ymax - ymin) * fraction_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _direct_total_component_arrays(
            concentration,
            LsT=float(params["LsT"]),
            Ns=float(params["Ns"]),
            Kds=float(params["Kds"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _basic_guess(compound)


class CompetitiveThreeStateSpecificKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model for specific binding."""

    name = "comp_3st_specific"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="RT",
            fitted_parameter="RT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kd",
            fitted_parameter="Kd",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("RT", min=0.0, vary=False),
        ParameterSpec("LsT", min=0.0, vary=False),
        ParameterSpec("Kds", min=0.0, vary=False),
        ParameterSpec("Kd", min=0.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        RT: float,
        LsT: float,
        Kds: float,
        Kd: float,
    ) -> np.ndarray:
        components = _competitive_three_state_specific_component_arrays(
            np.asarray(x, dtype=float),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            Kd=Kd,
        )
        fraction_tracer_bound = components["fraction_tracer_bound"]
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _competitive_three_state_specific_component_arrays(
            concentration,
            RT=float(params["RT"]),
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
            Kd=float(params["Kd"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)


class CompetitiveThreeStateTotalKdModel(BaseDoseResponseModel):
    """Three-state competitive-binding model with nonspecific binding."""

    name = "comp_3st_total"
    required_fixed_parameters = frozenset({"RT", "LsT", "Kds", "N"})
    concentration_parameters = frozenset({"RT", "LsT", "Kds", "Kd"})
    concentration_parameter_specs = (
        ConcentrationParameterSpec(
            parameter="RT",
            fitted_parameter="RT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="LsT",
            fitted_parameter="LsT",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kds",
            fitted_parameter="Kds",
            fitted_scale="linear",
            reportable=False,
        ),
        ConcentrationParameterSpec(
            parameter="Kd",
            fitted_parameter="Kd",
            fitted_scale="linear",
            reportable=True,
        ),
    )
    parameter_specs = (
        ParameterSpec("ymin"),
        ParameterSpec("ymax"),
        ParameterSpec("RT", min=0.0, vary=False),
        ParameterSpec("LsT", min=0.0, vary=False),
        ParameterSpec("Kds", min=0.0, vary=False),
        ParameterSpec("N", min=0.0, vary=False),
        ParameterSpec("Kd", min=0.0),
    )

    def evaluate(
        self,
        x: np.ndarray,
        *,
        ymin: float,
        ymax: float,
        RT: float,
        LsT: float,
        Kds: float,
        N: float,
        Kd: float,
    ) -> np.ndarray:
        components = _competitive_three_state_total_component_arrays(
            np.asarray(x, dtype=float),
            RT=RT,
            LsT=LsT,
            Kds=Kds,
            N=N,
            Kd=Kd,
        )
        fraction_tracer_bound = components["fraction_tracer_bound"]
        return ymin + (ymax - ymin) * fraction_tracer_bound

    def component_arrays(
        self,
        concentration: np.ndarray,
        x: np.ndarray,
        **params: float,
    ) -> dict[str, np.ndarray]:
        return _competitive_three_state_total_component_arrays(
            concentration,
            RT=float(params["RT"]),
            LsT=float(params["LsT"]),
            Kds=float(params["Kds"]),
            N=float(params["N"]),
            Kd=float(params["Kd"]),
        )

    def guess(self, compound: CompoundData) -> dict[str, float]:
        return _competition_guess(compound)
