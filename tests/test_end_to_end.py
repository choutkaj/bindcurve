"""Public workflows on exact data built independently of bindcurve's solvers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from references import MODEL_CASES, observations
from scipy.stats import t as student_t

import bindcurve as bc


def write_wide_csv(path, model, fixed, potencies=(1.0, 2.0, 4.0)):
    frames = []
    for k, potency in enumerate(potencies, start=1):
        x, y = observations(model, potency, fixed)
        frame = pd.DataFrame(
            {"compound_id": "a", "experiment_id": f"e{k}", "concentration": x}
        )
        # Symmetric replicate offsets with a different count per experiment.
        for replicate, offset in enumerate(np.linspace(-0.2, 0.2, k + 1), start=1):
            frame[f"response_{replicate}"] = y + offset
        frames.append(frame)
    pd.concat(frames).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("model", "parameter", "fixed"), MODEL_CASES, ids=[c[0] for c in MODEL_CASES]
)
def test_csv_to_report_and_plots(tmp_path, model, parameter, fixed):
    write_wide_csv(tmp_path / "data.csv", model, fixed)
    data = bc.DoseResponseData.from_csv(tmp_path / "data.csv", format="wide")
    results = bc.fit(data, model, fixed={"ymin": 0.0, "ymax": 100.0, **fixed})

    experiments = results.experiments()
    assert experiments["success"].all() and experiments["warnings"].isna().all()
    assert experiments["n_data"].tolist() == [16, 16, 16]
    np.testing.assert_allclose(experiments[parameter], [1.0, 2.0, 4.0], rtol=1e-6)

    row = results.summary().iloc[0]
    assert row[["N_fit", "N_failed", "N_flagged"]].tolist() == [3, 0, 0]
    assert row[parameter] == pytest.approx(2.0, rel=1e-6)
    # log10(1, 2, 4) has mean log10(2) and sample SD log10(2).
    assert row[f"log{parameter}_SD"] == pytest.approx(np.log10(2.0), rel=1e-6)
    factor = 2.0 ** (student_t.ppf(0.975, 2) / np.sqrt(3.0))
    np.testing.assert_allclose(
        row[[f"{parameter}_CI95_lower", f"{parameter}_CI95_upper"]].to_numpy(float),
        [2.0 / factor, 2.0 * factor],
        rtol=1e-6,
    )
    assert results.report(unit="uM").loc[0, "report"] == "2.0 [1.0, 4.0] uM"

    figure, ax = plt.subplots()
    bc.plot_fits(results, experiments="e2", errorbars=None, ax=ax)
    x, y = ax.lines[0].get_data()
    exact_x, exact_y = observations(model, 2.0, fixed)
    np.testing.assert_allclose(np.interp(exact_x, x, y), exact_y, atol=1e-3)
    bc.plot_compounds(results, ax=ax)
    bc.plot_residuals(results, ax=ax)
    figure.savefig(tmp_path / "plot.png")
    plt.close(figure)


def test_partial_failure_keeps_successful_experiments(tmp_path):
    write_wide_csv(tmp_path / "data.csv", "ic50", {}, potencies=(2.0, 8.0))
    table = pd.read_csv(tmp_path / "data.csv")
    short = pd.DataFrame(
        {
            "compound_id": ["a"],
            "experiment_id": ["short"],
            "concentration": [1.0],
            "response_1": [99.0],
        }
    )
    data = bc.DoseResponseData.from_wide(pd.concat([table, short]))
    results = bc.fit(
        data,
        "ic50",
        fixed={"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.0},
        errors="collect",
    )

    assert [fit.success for fit in results.fits] == [True, True, False]
    row = results.summary().iloc[0]
    assert row[["N_fit", "N_failed"]].tolist() == [2, 1]
    assert row["IC50"] == pytest.approx(4.0, rel=1e-6)

    # The compound curve uses the geometric-mean IC50, sqrt(2 * 8) = 4.
    figure, ax = plt.subplots()
    bc.plot_compounds(results, errorbars=None, ax=ax)
    x, y = ax.lines[-1].get_data()
    np.testing.assert_allclose(y, 100.0 / (1.0 + x / 4.0), rtol=1e-6)
    plt.close(figure)


def test_summary_converts_to_kd():
    x = np.logspace(-2, 2, 12)
    table = pd.concat(
        pd.DataFrame(
            {
                "compound_id": "a",
                "experiment_id": f"e{k}",
                "concentration": x,
                "response": 100.0 / (1.0 + x / ic50),
            }
        )
        for k, ic50 in enumerate((2.0, 4.0, 8.0))
    )
    results = bc.fit(
        bc.DoseResponseData(table),
        "ic50",
        fixed={"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.0},
    )
    summary = results.summary()
    kd = bc.cheng_prusoff(
        summary[["IC50", "IC50_CI95_lower", "IC50_CI95_upper"]], LsT=2.0, Kds=4.0
    )
    factor = 2.0 ** (student_t.ppf(0.975, 2) / np.sqrt(3.0))
    np.testing.assert_allclose(
        kd[0], [4.0 / 1.5, 4.0 / factor / 1.5, 4.0 * factor / 1.5], rtol=1e-6
    )
