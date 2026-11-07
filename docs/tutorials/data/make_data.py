"""Generate the synthetic tutorial datasets.

Run from the repository root with ``python docs/tutorials/data/make_data.py``.
Each dataset is drawn from a bindcurve model with the "true" parameters listed
below. Every compound has three independent experiments, which vary in their
parameters, and three technical replicates per concentration, which add
measurement noise. All concentrations are in nM. Output is deterministic for a
given seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bindcurve.models import get_model

SEED = 2026
N_EXPERIMENTS = 3
N_REPLICATES = 3
OUT = Path(__file__).parent

# IC50 assay: enzyme activity in % of uninhibited control.
# 11-point, 3-fold dilution series from 30 uM.
IC50_CONCENTRATIONS = 30_000.0 / 3.0 ** np.arange(11)
IC50_COMPOUNDS = {
    "inhibitor_a": {"IC50": 85.0, "hill_slope": 1.0},
    "inhibitor_b": {"IC50": 1_200.0, "hill_slope": 1.4},
}
IC50_PLATEAUS = {"ymin": 3.0, "ymax": 100.0}
IC50_NOISE = 2.0  # % activity, between technical replicates

# Fluorescence polarization assays, response in mP.
FP_PLATEAUS = {"ymin": 35.0, "ymax": 220.0}
FP_NOISE = 2.0  # mP, between technical replicates

# Direct binding: receptor titrated into fixed tracer.
# 12-point, 2-fold dilution series from 4 uM.
DIRECT_CONCENTRATIONS = 4_000.0 / 2.0 ** np.arange(12)
DIRECT_COMPOUNDS = {
    # Tracer far below its Kd: free and total receptor are nearly equal.
    "tracer_a": ("dir_specific", {"LsT": 1.0, "Kds": 40.0}),
    # Tracer comparable to its Kd: receptor depletion matters.
    "tracer_b": ("dir_specific", {"LsT": 10.0, "Kds": 15.0}),
    # Tracer that also binds nonspecifically.
    "tracer_c": ("dir_total", {"LsT": 10.0, "Ns": 0.25, "Kds": 60.0}),
}

# Competitive binding: competitor titrated into receptor and tracer_b.
# 12-point, 3-fold dilution series from 100 uM.
COMPETITION_CONCENTRATIONS = 100_000.0 / 3.0 ** np.arange(12)
COMPETITION_CONSTANTS = {"RT": 40.0, "LsT": 10.0, "Kds": 15.0}
COMPETITION_COMPOUNDS = {
    "competitor_a": ("comp_3st_specific", {"Kd": 50.0}),
    "competitor_b": ("comp_3st_specific", {"Kd": 800.0}),
    # Ternary complex with weaker tracer binding: displacement is incomplete.
    "competitor_c": ("comp_4st_specific", {"Kd": 100.0, "Kd3": 150.0}),
}

# Variation between independent experiments, which is larger than the noise
# between technical replicates. A well-run assay reproduces potency within a
# factor of 2-3, i.e. an SD of about 0.1-0.2 log units.
LOG10_SD = 0.1  # of fitted concentration parameters
PLATEAU_SD = 3.0  # response units


def experiments(rng, model_name, concentrations, true, plateaus, noise):
    """Rows of a wide table for one compound."""
    model = get_model(model_name)
    rows = []
    for e in range(1, N_EXPERIMENTS + 1):
        params = dict(true)
        for p in model.parameters:
            if p.concentration and not p.fixed and p.name in params:
                params[p.name] *= 10.0 ** rng.normal(0.0, LOG10_SD)
        for name, value in plateaus.items():
            params[name] = value + rng.normal(0.0, PLATEAU_SD)
        mean = model.evaluate(concentrations, **params)
        wells = mean[:, None] + rng.normal(0.0, noise, (len(mean), N_REPLICATES))
        for x, responses in zip(concentrations, wells):
            rows.append([f"exp_{e}", x, *responses])
    return rows


def write(path, blocks):
    columns = ["compound_id", "experiment_id", "concentration"] + [
        f"response_{i}" for i in range(1, N_REPLICATES + 1)
    ]
    rows = [[compound, *row] for compound, block in blocks for row in block]
    table = pd.DataFrame(rows, columns=columns)
    table["concentration"] = table["concentration"].map(lambda v: float(f"{v:.4g}"))
    for column in columns[3:]:
        table[column] = table[column].round(1)
    table.to_csv(path, index=False)


def main():
    rng = np.random.default_rng(SEED)
    write(
        OUT / "ic50.csv",
        [
            (name, experiments(rng, "ic50", IC50_CONCENTRATIONS, true,
                               IC50_PLATEAUS, IC50_NOISE))
            for name, true in IC50_COMPOUNDS.items()
        ],
    )
    write(
        OUT / "direct-binding.csv",
        [
            (name, experiments(rng, model, DIRECT_CONCENTRATIONS, true,
                               FP_PLATEAUS, FP_NOISE))
            for name, (model, true) in DIRECT_COMPOUNDS.items()
        ],
    )
    write(
        OUT / "competitive-binding.csv",
        [
            (name, experiments(rng, model, COMPETITION_CONCENTRATIONS,
                               {**COMPETITION_CONSTANTS, **true},
                               FP_PLATEAUS, FP_NOISE))
            for name, (model, true) in COMPETITION_COMPOUNDS.items()
        ],
    )


if __name__ == "__main__":
    main()
