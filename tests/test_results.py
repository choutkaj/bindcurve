import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

import bindcurve as bc


def ic50_data(potencies, compound_id="a"):
    x = np.logspace(-2, 2, 12)
    frames = [
        pd.DataFrame(
            {
                "compound_id": compound_id,
                "experiment_id": f"e{k}",
                "concentration": x,
                "response": 5.0 + 90.0 / (1.0 + (x / potency) ** 1.3) + np.sin(k + x),
            }
        )
        for k, potency in enumerate(potencies)
    ]
    return pd.concat(frames)


@pytest.fixture(scope="module")
def results():
    table = pd.concat([ic50_data([0.5, 1.0, 3.0]), ic50_data([2.0], "single")])
    return bc.fit(bc.DoseResponseData(table), "ic50")


def test_experiments_table_lists_estimates_and_diagnostics(results):
    table = results.experiments()
    assert len(table) == 4
    assert table["success"].all() and table["warnings"].isna().all()
    for name in ("ymin", "ymax", "IC50", "hill_slope"):
        assert table[name].notna().all() and (table[f"{name}_stderr"] > 0).all()


def test_summary_statistics_match_independent_calculation(results):
    estimates = results.experiments().query("compound_id == 'a'")
    row = results.summary().set_index("compound_id").loc["a"]
    log_ic50 = np.log10(estimates["IC50"].to_numpy())
    sem = log_ic50.std(ddof=1) / np.sqrt(3)
    t = student_t.ppf(0.975, 2)
    assert row["logIC50"] == pytest.approx(log_ic50.mean())
    assert row["IC50"] == pytest.approx(10 ** log_ic50.mean())
    assert row["logIC50_SD"] == pytest.approx(log_ic50.std(ddof=1))
    assert row["logIC50_SEM"] == pytest.approx(sem)
    assert row["IC50_CI95_lower"] == pytest.approx(10 ** (log_ic50.mean() - t * sem))
    assert row["IC50_CI95_upper"] == pytest.approx(10 ** (log_ic50.mean() + t * sem))
    hill = estimates["hill_slope"].to_numpy()
    assert row["hill_slope"] == pytest.approx(hill.mean())
    assert row["hill_slope_SD"] == pytest.approx(hill.std(ddof=1))
    assert row["hill_slope_CI95_upper"] == pytest.approx(
        hill.mean() + t * hill.std(ddof=1) / np.sqrt(3)
    )
    assert row[["N_fit", "N_failed", "N_flagged"]].tolist() == [3, 0, 0]


def test_single_experiment_has_no_spread(results):
    row = results.summary().set_index("compound_id").loc["single"]
    assert row["N_fit"] == 1 and np.isfinite(row["IC50"])
    assert np.isnan(row["logIC50_SD"]) and np.isnan(row["IC50_CI95_lower"])


def test_parameters_combine_summary_centers_and_fixed_values():
    data = bc.DoseResponseData(ic50_data([0.5, 2.0]))
    results = bc.fit(data, "ic50", fixed={"ymin": 5.0, "ymax": 95.0})
    values = results.parameters("a")
    assert values["ymin"] == 5.0 and values["ymax"] == 95.0
    assert values["IC50"] == pytest.approx(results.summary().loc[0, "IC50"])
    assert set(values) == {"ymin", "ymax", "IC50", "hill_slope"}


def test_report_formats(results):
    summary = results.summary().set_index("compound_id")
    row = summary.loc["a"]
    center, log_mean, log_sd = row["IC50"], row["logIC50"], row["logIC50_SD"]
    report = results.report(unit="uM").set_index("compound_id")
    low, high = 10 ** (log_mean - log_sd), 10 ** (log_mean + log_sd)
    # "#.2g" keeps trailing zeros: two significant figures for values in [0.1, 10).
    assert report.loc["a", "report"] == f"{center:#.2g} [{low:#.2g}, {high:#.2g}] uM"
    assert report.loc["single", "report"] == f"{summary.loc['single', 'IC50']:#.2g} uM"
    log_report = results.report(log=True).set_index("compound_id")
    assert log_report.loc["a", "report"] == f"{log_mean:.2f} ± {log_sd:.2f}"
    ci = results.report(uncertainty="ci95", log=True, digits=3).set_index("compound_id")
    lower, upper = np.log10(row[["IC50_CI95_lower", "IC50_CI95_upper"]])
    assert ci.loc["a", "report"] == f"{log_mean:.3f} [{lower:.3f}, {upper:.3f}]"
    assert list(report.columns) == ["report", "N_fit", "N_failed", "N_flagged"]


def test_report_rejects_invalid_requests(results):
    with pytest.raises(ValueError, match="uncertainty"):
        results.report(uncertainty="range")
    with pytest.raises(ValueError, match=r"\['IC50'\]"):
        results.report("hill_slope")


def test_failed_compounds_are_reported_as_unavailable():
    x = np.logspace(-2, 2, 6)
    table = pd.DataFrame(
        {"compound_id": "a", "concentration": x[:2], "response": [90.0, 10.0]}
    )
    results = bc.fit(bc.DoseResponseData(table), "ic50", errors="collect")
    assert results.report().loc[0, "report"] == "no successful fit"
    assert np.isnan(results.summary().loc[0, "IC50"])
    with pytest.raises(ValueError, match="no successful fits"):
        results.parameters("a")
