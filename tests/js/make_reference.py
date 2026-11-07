"""Write reference values of every bindcurve model for the Playground JS test.

Usage: ``python tests/js/make_reference.py reference.json``. The JSON holds, for
each model and parameter set, the concentrations and the Python results of
``Model.evaluate`` and, for binding models, ``BindingModel.species``.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from bindcurve.models import BindingModel, get_model

X = np.concatenate([[1e-6], np.logspace(-3, 6, 61)])
FP = {"ymin": 35.0, "ymax": 220.0}

CASES = {
    "ic50": [
        {"ymin": 0.0, "ymax": 100.0, "IC50": 100.0, "hill_slope": 1.0},
        {"ymin": 95.0, "ymax": 5.0, "IC50": 0.02, "hill_slope": 3.5},
        {"ymin": -10.0, "ymax": 120.0, "IC50": 5e4, "hill_slope": 0.3},
    ],
    "dir_simple": [{**FP, "Kds": 15.0}, {**FP, "Kds": 1e-2}, {**FP, "Kds": 1e5}],
    "dir_specific": [
        {**FP, "LsT": 10.0, "Kds": 15.0},
        {**FP, "LsT": 1000.0, "Kds": 1.0},
        {**FP, "LsT": 0.01, "Kds": 500.0},
    ],
    "dir_total": [
        {**FP, "LsT": 10.0, "Ns": 0.25, "Kds": 60.0},
        {**FP, "LsT": 500.0, "Ns": 4.0, "Kds": 0.5},
    ],
    "comp_3st_specific": [
        {**FP, "RT": 40.0, "LsT": 10.0, "Kds": 15.0, "Kd": 50.0},
        {**FP, "RT": 1000.0, "LsT": 1.0, "Kds": 0.1, "Kd": 0.01},
        {**FP, "RT": 0.5, "LsT": 100.0, "Kds": 300.0, "Kd": 1e4},
    ],
    "comp_3st_total": [
        {**FP, "RT": 40.0, "LsT": 10.0, "Kds": 15.0, "N": 0.5, "Kd": 50.0},
        {**FP, "RT": 200.0, "LsT": 50.0, "Kds": 2.0, "N": 4.0, "Kd": 0.3},
    ],
    "comp_4st_specific": [
        {**FP, "RT": 40.0, "LsT": 10.0, "Kds": 15.0, "Kd3": 150.0, "Kd": 50.0},
        {**FP, "RT": 40.0, "LsT": 10.0, "Kds": 15.0, "Kd3": 1.0, "Kd": 50.0},
        {**FP, "RT": 500.0, "LsT": 200.0, "Kds": 5.0, "Kd3": 1e5, "Kd": 0.5},
        {**FP, "RT": 2.0, "LsT": 0.5, "Kds": 100.0, "Kd3": 100.0, "Kd": 3.0},
    ],
    "comp_4st_total": [
        {
            **FP,
            "RT": 40.0,
            "LsT": 10.0,
            "Kds": 15.0,
            "Kd3": 150.0,
            "N": 0.5,
            "Kd": 50.0,
        },
        {**FP, "RT": 300.0, "LsT": 20.0, "Kds": 1.0, "Kd3": 0.2, "N": 3.0, "Kd": 10.0},
    ],
}


def main(path: str) -> None:
    cases = []
    for name, parameter_sets in CASES.items():
        model = get_model(name)
        for params in parameter_sets:
            case = {
                "model": name,
                "params": params,
                "x": X.tolist(),
                "response": model.evaluate(X, **params).tolist(),
            }
            if isinstance(model, BindingModel):
                case["species"] = {
                    key: np.broadcast_to(value, X.shape).tolist()
                    for key, value in model.species(X, **params).items()
                }
            cases.append(case)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(cases, file)


if __name__ == "__main__":
    main(sys.argv[1])
