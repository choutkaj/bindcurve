"""Public workflows with deterministic, independent scientific expectations.

Inputs come from logistic equations or free-species mass balances, never from
bindcurve's model evaluators. Numerical tolerances allow optimizer/platform
variation; CSV values and plotted coordinates matter, not PNG bytes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import t as student_t

import bindcurve as bc

IC50_FIXED = {"ymin": 0.0, "ymax": 100.0, "hill_slope": 1.0}
COMPETITIVE_FIXED = {"ymin": 0.0, "ymax": 100.0, "RT": 0.05, "LsT": 0.005, "Kds": 0.02}
MODEL_CASES = [
    ("ic50", "IC50", IC50_FIXED),
    ("dir_simple", "Kds", {"ymin": 0.0, "ymax": 100.0}),
    ("dir_specific", "Kds", {"ymin": 0.0, "ymax": 100.0, "LsT": 0.4}),
    ("dir_total", "Kds", {"ymin": 0.0, "ymax": 100.0, "LsT": 0.4, "Ns": 0.25}),
    ("comp_3st_specific", "Kd", COMPETITIVE_FIXED),
    ("comp_3st_total", "Kd", {**COMPETITIVE_FIXED, "N": 0.35}),
    ("comp_4st_specific", "Kd", {**COMPETITIVE_FIXED, "Kd3": 0.5}),
    ("comp_4st_total", "Kd", {**COMPETITIVE_FIXED, "Kd3": 0.5, "N": 0.35}),
]


def reference_observations(model, potency, fixed):
    """Construct total concentrations from free species and their complexes.

    Competitive equilibria obey RL = R*L/Kd, RLs = R*Ls/Kds and, for four
    states, RLLs = R*L*Ls/(Kd*Kd3). Choosing free L first avoids the production
    solvers' inverse problem of finding species from total competitor.
    """
    free = np.logspace(-3, 2, 16)
    if model == "ic50":
        return free, 100.0 / (1.0 + free / potency)
    if model.startswith("dir_"):
        fraction = free / ((1.0 + fixed.get("Ns", 0.0)) * potency + free)
        total = free + fixed.get("LsT", 0.0) * fraction
        return total, 100.0 * fraction

    RT, LsT, Kds = fixed["RT"], fixed["LsT"], fixed["Kds"]
    concentrations, responses = [], []
    for L in free:
        ternary_factor = L / (potency * fixed["Kd3"]) if "Kd3" in fixed else 0.0
        tracer_factor = 1.0 / Kds + ternary_factor
        R = brentq(
            lambda R, L=L, tracer_factor=tracer_factor: (
                R * (1.0 + L / potency)
                + LsT * R * tracer_factor / (1.0 + R * tracer_factor)
                - RT
            ),
            0.0,
            RT,
            xtol=1e-15,
        )
        Ls = LsT / (1.0 + R * tracer_factor)
        RL, RLs, RLLs = R * L / potency, R * Ls / Kds, R * Ls * ternary_factor
        concentrations.append((1.0 + fixed.get("N", 0.0)) * L + RL + RLLs)
        responses.append(100.0 * (RLs + RLLs) / LsT)
    return np.array(concentrations), np.array(responses)


def write_wide_input(path, model, fixed, potencies=(1.0, 2.0, 4.0)):
    frames = []
    for index, potency in enumerate(potencies, start=1):
        x, y = reference_observations(model, potency, fixed)
        # Different replicate counts must not change experiment-level weighting.
        frame = pd.DataFrame(
            {
                "compound_id": "cmpd_a",
                "experiment_id": f"exp{index}",
                "concentration": x,
            }
        )
        for replicate, offset in enumerate(np.linspace(-0.2, 0.2, index + 1), start=1):
            frame[f"response_{replicate}"] = y + offset
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def assert_csv_round_trip(table, path):
    table.to_csv(path, index=False)
    # CSV has no dtype metadata and represents missing diagnostics as empty cells.
    pd.testing.assert_frame_equal(
        pd.read_csv(path).fillna(""),
        table.reset_index(drop=True).fillna(""),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "model,parameter,fixed", MODEL_CASES, ids=[c[0] for c in MODEL_CASES]
)
def test_file_to_fit_summary_report_and_plot(tmp_path, ax, model, parameter, fixed):
    input_path = tmp_path / "observations.csv"
    write_wide_input(input_path, model, fixed)
    data = bc.DoseResponseData.from_csv(input_path, format="wide")
    results = bc.fit(data, model=model, fixed=fixed)

    fits = results.fit_summary().sort_values("experiment_id")
    assert fits["success"].tolist() == [True, True, True]
    assert fits["n_data"].tolist() == [16, 16, 16]
    np.testing.assert_allclose(fits[parameter], [1.0, 2.0, 4.0], rtol=1e-5)
    for name, value in fixed.items():
        np.testing.assert_allclose(fits[name], value)

    summary = results.summary()
    row = summary.iloc[0]
    assert (
        row["N_exp"],
        row["N_fit_successful"],
        row["N_fit_failed"],
        row["N_obs"],
    ) == (3, 3, 0, 48)
    assert row[parameter] == pytest.approx(2.0, rel=1e-5)
    # log10(1, 2, 4) has sample SD log10(2), independently of replicate counts.
    ci_factor = 2.0 ** (student_t.ppf(0.975, df=2) / np.sqrt(3.0))
    np.testing.assert_allclose(
        row[
            [
                f"{parameter}_SD_lower",
                f"{parameter}_SD_upper",
                f"{parameter}_CI95_lower",
                f"{parameter}_CI95_upper",
            ]
        ].to_numpy(dtype=float),
        [1.0, 4.0, 2.0 / ci_factor, 2.0 * ci_factor],
        rtol=1e-5,
    )
    report = results.report(rounding="decimals", places_mean=2, unit="uM")
    assert "2.00" in report.loc[0, "report"]
    assert "uM" in report.loc[0, "report"]
    for name, table in [("fits", fits), ("summary", summary), ("report", report)]:
        assert_csv_round_trip(table, tmp_path / f"{name}.csv")

    x, expected_y = reference_observations(model, 2.0, fixed)
    bc.plot_fits(
        data,
        results,
        ax=ax,
        experiments=["exp2"],
        x_grid=x,
        show_markers=False,
        show_errorbars=False,
    )
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), x)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_y, atol=1e-5)
    image_path = tmp_path / "fits.png"
    ax.figure.savefig(image_path)
    pixels = plt.imread(image_path)
    assert pixels.ndim == 3 and np.ptp(pixels) > 0


@pytest.mark.parametrize("uncertainty", ["sigma", "weight"])
def test_weighted_file_to_fit_residuals_and_export(tmp_path, ax, uncertainty):
    x = np.logspace(-2, 2, 12)
    y = 100.0 / (1.0 + x / 2.0) + np.sin(np.arange(len(x)))
    sigma = np.column_stack(
        (np.linspace(0.4, 1.2, len(x)), np.linspace(1.5, 3.0, len(x)))
    )
    responses = np.column_stack((y - 0.3, y + 0.3))
    table = pd.DataFrame(
        {
            "compound_id": "cmpd_a",
            "experiment_id": "exp1",
            "concentration": np.repeat(x, 2),
            "replicate_id": ["rep1", "rep2"] * len(x),
            "response": responses.ravel(),
            uncertainty: (sigma if uncertainty == "sigma" else 1.0 / sigma).ravel(),
        }
    )
    path = tmp_path / "weighted.csv"
    table.to_csv(path, index=False)
    data = bc.DoseResponseData.from_csv(path)
    results = bc.fit(data, model="ic50", fixed=IC50_FIXED)

    # Arithmetic replicate means; their sigma is sqrt(sum(sigma_i**2))/n.
    sigma_mean = np.sqrt(np.sum(sigma**2, axis=1)) / 2.0
    optimum = minimize_scalar(
        lambda log_ic50: np.sum(
            ((y - 100.0 / (1.0 + x / 10**log_ic50)) / sigma_mean) ** 2
        ),
        bounds=(-1.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert optimum.success
    fits = results.fit_summary()
    fit = fits.iloc[0]
    assert fit["success"]
    assert fit["IC50"] == pytest.approx(10**optimum.x, rel=1e-5)
    residuals = y - 100.0 / (1.0 + x / fit["IC50"])
    standardized = residuals / sigma_mean
    assert fit["n_data"] == len(x)
    assert fit["chi_square"] == pytest.approx(np.sum(standardized**2))
    assert fit["reduced_chi_square"] == pytest.approx(
        np.sum(standardized**2) / (len(x) - 1)
    )
    assert fit["rss"] == pytest.approx(np.sum(residuals**2))
    bc.plot_residuals(data, results, ax=ax, standardized=True)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(), np.column_stack((x, standardized))
    )

    assert_csv_round_trip(fits, tmp_path / "weighted_fits.csv")
    data.to_json(tmp_path / "weighted_data.json")
    restored = bc.DoseResponseData.from_json(tmp_path / "weighted_data.json")
    restored_fits = bc.fit(restored, model="ic50", fixed=IC50_FIXED).fit_summary()
    np.testing.assert_allclose(
        restored_fits[["IC50", "chi_square"]], fits[["IC50", "chi_square"]], rtol=1e-5
    )


def test_partial_failure_preserves_successful_analysis(tmp_path, ax):
    path = tmp_path / "partial.csv"
    write_wide_input(path, "ic50", IC50_FIXED, potencies=(2.0, 8.0))
    table = pd.read_csv(path)
    # One distinct concentration cannot fit one varying parameter.
    failed = pd.DataFrame(
        {
            "compound_id": ["cmpd_a"],
            "experiment_id": ["too_short"],
            "concentration": [1.0],
            "response_1": [99.0],
        }
    )
    pd.concat([table, failed], ignore_index=True).to_csv(path, index=False)
    data = bc.DoseResponseData.from_csv(path, format="wide")
    results = bc.fit(
        data, model="ic50", fixed=IC50_FIXED, settings=bc.FitSettings(errors="collect")
    )
    fits = results.fit_summary().set_index("experiment_id")
    assert fits["success"].to_dict() == {"exp1": True, "exp2": True, "too_short": False}
    assert fits.loc["too_short", "error_type"] == "ValueError"
    assert pd.isna(fits.loc["too_short", "IC50"])
    row = results.summary().iloc[0]
    assert (
        row["N_exp"],
        row["N_fit_successful"],
        row["N_fit_failed"],
        row["N_obs"],
    ) == (3, 2, 1, 32)
    assert row["IC50"] == pytest.approx(4.0, rel=1e-5)
    report = results.report()
    assert report.loc[0, "N_fit_successful"] == 2
    assert report.loc[0, "N_fit_failed"] == 1
    assert_csv_round_trip(fits.reset_index(), tmp_path / "partial_fits.csv")

    x = np.logspace(-2, 2, 25)
    bc.plot_compounds(
        data, results, ax=ax, x_grid=x, show_markers=False, show_errorbars=False
    )
    assert len(ax.lines) == 1
    expected = (100.0 / (1.0 + x / 2.0) + 100.0 / (1.0 + x / 8.0)) / 2.0
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected, atol=1e-5)


def test_actual_fit_summary_to_conversion_and_export(tmp_path):
    path = tmp_path / "ic50.csv"
    write_wide_input(path, "ic50", IC50_FIXED, potencies=(2.0, 4.0, 8.0))
    data = bc.DoseResponseData.from_csv(path, format="wide")
    results = bc.fit(data, model="ic50", fixed=IC50_FIXED)
    converted = bc.convert_ic50_to_kd(
        results.summary(),
        model="cheng_prusoff",
        LsT=2.0,
        Kds=4.0,
        lower_col="IC50_CI95_lower",
        upper_col="IC50_CI95_upper",
    )
    converted.to_csv(tmp_path / "converted.csv", index=False)
    saved = pd.read_csv(tmp_path / "converted.csv")
    assert saved["compound_id"].tolist() == ["cmpd_a"]
    assert saved["model"].tolist() == ["cheng_prusoff"]
    factor = 2.0 ** (student_t.ppf(0.975, df=2) / np.sqrt(3.0))
    np.testing.assert_allclose(
        saved[["Kd", "lower_Kd", "upper_Kd"]],
        [[4.0 / 1.5, 4.0 / factor / 1.5, 4.0 * factor / 1.5]],
        rtol=1e-5,
    )
