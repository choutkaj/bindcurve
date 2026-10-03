"""Shared input construction; scientific reference curves stay in their tests."""

import numpy as np
import pandas as pd
import pytest

import bindcurve as bc


@pytest.fixture
def make_competition_data():
    def build(curve, *, compound_id="cmpd_a") -> bc.DoseResponseData:
        concentrations = np.logspace(-3, 2, 22)
        rows = []
        multipliers = {"exp1": 0.95, "exp2": 1.00, "exp3": 1.05}
        for experiment_id, multiplier in multipliers.items():
            for concentration in concentrations:
                response = curve(concentration * multiplier)
                for replicate_id, noise in enumerate([-0.08, 0.0, 0.08], start=1):
                    rows.append(
                        {
                            "compound_id": compound_id,
                            "experiment_id": experiment_id,
                            "concentration": concentration,
                            "replicate_id": f"rep{replicate_id}",
                            "response": response + noise,
                        }
                    )
        return bc.DoseResponseData.from_dataframe(
            pd.DataFrame(rows),
        )

    return build
