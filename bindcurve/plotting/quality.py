from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from bindcurve.datasets import DoseResponseData
from bindcurve.modeling import get_model
from bindcurve.quality import (
    DataQualityThresholds,
    ResultQualityThresholds,
    resolve_requested_compounds,
)
from bindcurve.results import ConcentrationSummary, FitResult, FitResults

STATUS_COLORS = {
    "green": "#3c8d40",
    "orange": "#d18b15",
    "red": "#c64639",
}
STATUS_FILLS = {
    "green": "#ebf5ec",
    "orange": "#fcf2df",
    "red": "#fae8e6",
}
STATUS_CMAP = LinearSegmentedColormap.from_list(
    "bindcurve_status",
    [STATUS_COLORS["green"], STATUS_COLORS["orange"], STATUS_COLORS["red"]],
)


@dataclass(frozen=True)
class DataDashboardDetails:
    compound_id: str
    experiments: list[str]
    concentrations: list[float]
    replicate_counts: np.ndarray
    noise_frac_range: np.ndarray


@dataclass(frozen=True)
class ResultDashboardDetails:
    compound_id: str
    parameter: str
    log_parameter: str
    fit_table: pd.DataFrame
    log_values: pd.DataFrame
    concentration_summary: ConcentrationSummary | None


def plot_data_quality_dashboard(
    data: DoseResponseData,
    *,
    compounds: str | list[str] | None = None,
    thresholds: DataQualityThresholds | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a graphical dashboard summarizing data-level QC."""
    resolved_thresholds = thresholds or DataQualityThresholds()
    selected_compounds = resolve_requested_compounds(data.compounds, compounds)
    report = data.quality_report(
        compounds=selected_compounds,
        thresholds=resolved_thresholds,
    )
    if report.empty:
        figure = plt.figure(figsize=figsize or (10.0, 3.0), constrained_layout=True)
        figure.suptitle("Data Quality Dashboard", fontsize=14, fontweight="bold")
        plt.close(figure)
        return figure

    n_compounds = len(selected_compounds)
    if figsize is None:
        figsize = (15.0, max(2.6 * n_compounds, 2.6))
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(
            h_pad=0.02,
            w_pad=0.02,
            hspace=0.02,
            wspace=0.03,
        )
    figure.suptitle("Data Quality Dashboard", fontsize=14, fontweight="bold")
    grid = figure.add_gridspec(
        n_compounds,
        3,
        width_ratios=[1.2, 1.3, 1.5],
    )

    for row_index, compound_id in enumerate(selected_compounds):
        compound_table = data.select_compound(compound_id).table
        report_row = report.loc[report["compound_id"] == compound_id].iloc[0]
        details = _build_data_dashboard_details(compound_table)

        summary_ax = figure.add_subplot(grid[row_index, 0])
        count_ax = figure.add_subplot(grid[row_index, 1])
        noise_ax = figure.add_subplot(grid[row_index, 2])

        _draw_summary_card(
            summary_ax,
            title=f"{compound_id} - data QC",
            status=str(report_row["status"]),
            lines=[
                f"N_exp: {int(report_row['N_exp'])}",
                f"N_obs: {int(report_row['N_obs'])}",
                f"Grid coverage: {_format_float(report_row['grid_coverage'], 2)}",
                (
                    "Replicates/cell min-med-max: "
                    f"{_format_int(report_row['N_rep_min'])} / "
                    f"{_format_float(report_row['N_rep_median'], 1)} / "
                    f"{_format_int(report_row['N_rep_max'])}"
                ),
                (
                    "Single-replicate fraction: "
                    f"{_format_float(report_row['single_replicate_fraction'], 2)}"
                ),
                (
                    "Median noise frac-range: "
                    f"{_format_float(report_row['intra_noise_median_frac_range'], 2)}"
                ),
                (
                    "P90 noise frac-range: "
                    f"{_format_float(report_row['intra_noise_p90_frac_range'], 2)}"
                ),
                f"Flags: {_format_flags(report_row['flags'])}",
            ],
        )
        _plot_matrix(
            count_ax,
            matrix=details.replicate_counts,
            x_labels=_format_concentration_labels(details.concentrations),
            y_labels=details.experiments,
            title="Replicate Count",
            cmap="Blues",
            vmin=0.0,
            vmax=max(1.0, float(np.nanmax(details.replicate_counts))),
            value_format="{:.0f}",
        )
        _plot_matrix(
            noise_ax,
            matrix=details.noise_frac_range,
            x_labels=_format_concentration_labels(details.concentrations),
            y_labels=details.experiments,
            title="Replicate SD / Response Range",
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=max(
                resolved_thresholds.max_intra_noise_p90_frac_range_red,
                _safe_nanmax(details.noise_frac_range, fallback=0.0),
            ),
            value_format="{:.2f}",
        )

    plt.close(figure)
    return figure


def plot_results_quality_dashboard(
    results: FitResults,
    *,
    parameter: str = "auto",
    compounds: str | list[str] | None = None,
    thresholds: ResultQualityThresholds | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a graphical dashboard summarizing fit- and summary-level QC."""
    available_compounds = _ordered_result_compounds(results)
    selected_compounds = resolve_requested_compounds(available_compounds, compounds)
    resolved_thresholds = thresholds or ResultQualityThresholds()
    report = results.quality_report(
        parameter=parameter,
        compounds=selected_compounds,
        thresholds=resolved_thresholds,
    )
    if report.empty:
        figure = plt.figure(figsize=figsize or (10.0, 3.0), constrained_layout=True)
        figure.suptitle("Results Quality Dashboard", fontsize=14, fontweight="bold")
        plt.close(figure)
        return figure

    resolved_parameter = str(report.iloc[0]["parameter"])
    n_compounds = len(selected_compounds)
    if figsize is None:
        figsize = (16.0, max(2.9 * n_compounds, 2.9))
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(
            h_pad=0.02,
            w_pad=0.02,
            hspace=0.02,
            wspace=0.03,
        )
    figure.suptitle("Results Quality Dashboard", fontsize=14, fontweight="bold")
    grid = figure.add_gridspec(
        n_compounds,
        3,
        width_ratios=[1.2, 1.2, 1.8],
    )

    for row_index, compound_id in enumerate(selected_compounds):
        report_row = report.loc[report["compound_id"] == compound_id].iloc[0]
        details = _build_result_dashboard_details(
            results,
            compound_id=compound_id,
            parameter=resolved_parameter,
            thresholds=resolved_thresholds,
        )
        total_fits = int(report_row["N_fit_success"]) + int(report_row["N_fit_failed"])

        summary_ax = figure.add_subplot(grid[row_index, 0])
        estimate_ax = figure.add_subplot(grid[row_index, 1])
        table_ax = figure.add_subplot(grid[row_index, 2])

        _draw_summary_card(
            summary_ax,
            title=f"{compound_id} - results QC",
            status=str(report_row["status"]),
            lines=[
                f"Parameter: {report_row['parameter']}",
                f"N_exp: {int(report_row['N_exp'])}",
                (
                    "Successful fits: "
                    f"{int(report_row['N_fit_success'])} / "
                    f"{total_fits}"
                ),
                (
                    "R_squared median/min: "
                    f"{_format_float(report_row['R_squared_median'], 3)} / "
                    f"{_format_float(report_row['R_squared_min'], 3)}"
                ),
                f"redchi median: {_format_float(report_row['redchi_median'], 3)}",
                (
                    "inter log10 SD: "
                    f"{_format_float(report_row['inter_log10_sd'], 3)}"
                ),
                (
                    "inter CI95 fold: "
                    f"{_format_float(report_row['inter_ci95_fold'], 2)}"
                ),
                f"Flags: {_format_flags(report_row['flags'])}",
            ],
        )
        _plot_log_estimate_panel(
            estimate_ax,
            details=details,
        )
        _plot_fit_diagnostic_table(
            table_ax,
            fit_table=details.fit_table,
        )

    plt.close(figure)
    return figure


def _build_data_dashboard_details(compound_table: pd.DataFrame) -> DataDashboardDetails:
    compound_id = str(compound_table["compound_id"].iloc[0])
    experiments = sorted(compound_table["experiment_id"].astype(str).unique())
    concentrations = sorted(compound_table["concentration"].unique())
    experiment_index = {
        experiment: index for index, experiment in enumerate(experiments)
    }
    concentration_index = {
        float(concentration): index
        for index, concentration in enumerate(concentrations)
    }
    replicate_counts = np.full((len(experiments), len(concentrations)), np.nan)
    noise_frac_range = np.full((len(experiments), len(concentrations)), np.nan)

    cell_stats = compound_table.groupby(
        ["experiment_id", "concentration"],
        as_index=False,
    )["response"].agg(response_sd="std", n_replicates="count")
    experiment_means = (
        compound_table.groupby(
            ["experiment_id", "concentration"],
            as_index=False,
        )["response"]
        .mean()
        .rename(columns={"response": "response_mean"})
    )
    experiment_ranges = (
        experiment_means.groupby("experiment_id")["response_mean"]
        .agg(lambda values: float(values.max() - values.min()))
        .rename("experiment_response_range")
        .reset_index()
    )
    cell_stats = cell_stats.merge(
        experiment_ranges,
        on="experiment_id",
        how="left",
    )

    for row in cell_stats.itertuples():
        y_index = experiment_index[str(row.experiment_id)]
        x_index = concentration_index[float(row.concentration)]
        replicate_counts[y_index, x_index] = float(row.n_replicates)
        if (
            row.n_replicates >= 2
            and pd.notna(row.response_sd)
            and row.experiment_response_range > 0.0
        ):
            noise_frac_range[y_index, x_index] = float(
                row.response_sd / row.experiment_response_range
            )

    return DataDashboardDetails(
        compound_id=compound_id,
        experiments=experiments,
        concentrations=[float(value) for value in concentrations],
        replicate_counts=replicate_counts,
        noise_frac_range=noise_frac_range,
    )


def _build_result_dashboard_details(
    results: FitResults,
    *,
    compound_id: str,
    parameter: str,
    thresholds: ResultQualityThresholds,
) -> ResultDashboardDetails:
    compound_fits = [
        fit for fit in results.fit_results if str(fit.compound_id) == str(compound_id)
    ]
    concentration_summary = next(
        (
            summary
            for summary in results.summaries
            if isinstance(summary, ConcentrationSummary)
            and summary.compound_id == compound_id
            and summary.parameter == parameter
        ),
        None,
    )
    log_parameter = (
        concentration_summary.log_parameter
        if concentration_summary is not None
        else f"log{parameter}"
    )

    fit_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    for fit in compound_fits:
        covariance_missing = (
            fit.lmfit_result is None or getattr(fit.lmfit_result, "covar", None) is None
        )
        stderr_missing = _fit_has_missing_stderr(fit)
        parameter_at_bound = _fit_has_parameter_at_bound(
            fit,
            rel_tol=thresholds.bound_tolerance_rel,
            abs_tol=thresholds.bound_tolerance_abs,
        )
        parameter_stderr = _fit_parameter_stderr(fit, parameter=parameter)
        parameter_bound_gap = _fit_parameter_bound_gap(fit, parameter=parameter)
        fit_rows.append(
            {
                "experiment_id": fit.experiment_id or fit.model_name,
                "success": "yes" if fit.success else "no",
                "r_squared": (
                    np.nan
                    if fit.metrics is None or fit.metrics.r_squared is None
                    else float(fit.metrics.r_squared)
                ),
                "redchi": (
                    np.nan if fit.metrics is None else float(fit.metrics.redchi)
                ),
                "covariance": (
                    "NA"
                    if not fit.success
                    else ("missing" if covariance_missing else "yes")
                ),
                "stderr": parameter_stderr,
                "bound_gap": parameter_bound_gap,
                "fit_status": _fit_status(
                    fit=fit,
                    covariance_missing=covariance_missing,
                    stderr_missing=stderr_missing,
                    parameter_at_bound=parameter_at_bound,
                ),
            }
        )
        log_value = _fit_log10_value(fit, parameter=parameter)
        if log_value is not None:
            log_rows.append(
                {
                    "experiment_id": fit.experiment_id or fit.model_name,
                    "log10_value": log_value,
                    "fit_status": _fit_status(
                        fit=fit,
                        covariance_missing=covariance_missing,
                        stderr_missing=stderr_missing,
                        parameter_at_bound=parameter_at_bound,
                    ),
                }
            )

    return ResultDashboardDetails(
        compound_id=compound_id,
        parameter=parameter,
        log_parameter=log_parameter,
        fit_table=pd.DataFrame(fit_rows),
        log_values=pd.DataFrame(log_rows),
        concentration_summary=concentration_summary,
    )


def _draw_summary_card(
    ax: Axes,
    *,
    title: str,
    status: str,
    lines: list[str],
) -> None:
    ax.set_title(f"{title} ({status.upper()})", loc="left", fontweight="bold")
    ax.set_facecolor(STATUS_FILLS.get(status, "#f5f5f5"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(STATUS_COLORS.get(status, "#808080"))
        spine.set_linewidth(1.2)
    body = "\n".join(textwrap.fill(line, width=36) for line in lines)
    ax.text(
        0.03,
        0.97,
        body,
        va="top",
        ha="left",
        fontsize=8,
        transform=ax.transAxes,
    )


def _plot_matrix(
    ax: Axes,
    *,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    value_format: str,
) -> None:
    plot_matrix = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(plot_matrix)
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    image.cmap.set_bad("#f2f2f2")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("concentration")
    ax.set_ylabel("experiment")

    for y_index in range(plot_matrix.shape[0]):
        for x_index in range(plot_matrix.shape[1]):
            value = plot_matrix[y_index, x_index]
            if np.isnan(value):
                continue
            text_color = (
                "white"
                if _relative_intensity(value, vmin, vmax) > 0.55
                else "black"
            )
            ax.text(
                x_index,
                y_index,
                value_format.format(value),
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)


def _plot_log_estimate_panel(
    ax: Axes,
    *,
    details: ResultDashboardDetails,
) -> None:
    ax.set_title(f"Per-experiment {details.log_parameter}", fontweight="bold")
    if details.log_values.empty:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No successful concentration estimates",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    x_positions = np.arange(len(details.log_values))
    colors = [
        STATUS_COLORS.get(str(status), "#808080")
        for status in details.log_values["fit_status"]
    ]
    ax.scatter(
        x_positions,
        details.log_values["log10_value"].to_numpy(dtype=float),
        s=42,
        c=colors,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    summary = details.concentration_summary
    if summary is not None:
        if (
            summary.log10_ci95_lower is not None
            and summary.log10_ci95_upper is not None
        ):
            ax.axhspan(
                summary.log10_ci95_lower,
                summary.log10_ci95_upper,
                color="#cfd8e3",
                alpha=0.45,
                zorder=1,
            )
        ax.axhline(
            summary.log10_mean,
            color="#2f2f2f",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(details.log_values["experiment_id"], rotation=45, ha="right")
    ax.set_ylabel(details.log_parameter)
    ax.set_xlabel("experiment")


def _plot_fit_diagnostic_table(
    ax: Axes,
    *,
    fit_table: pd.DataFrame,
) -> None:
    ax.set_title("Per-fit diagnostics", fontweight="bold")
    ax.axis("off")
    if fit_table.empty:
        ax.text(0.5, 0.5, "No fit diagnostics available", ha="center", va="center")
        return

    display = fit_table.copy()
    display["r_squared"] = display["r_squared"].apply(
        lambda value: _format_float(value, 3)
    )
    display["redchi"] = display["redchi"].apply(lambda value: _format_float(value, 3))
    display["stderr"] = display["stderr"].apply(_format_diagnostic_value)
    display["bound_gap"] = display["bound_gap"].apply(_format_diagnostic_value)
    display = display[
        [
            "experiment_id",
            "success",
            "r_squared",
            "redchi",
            "covariance",
            "stderr",
            "bound_gap",
        ]
    ]
    table = ax.table(
        cellText=display.to_numpy().tolist(),
        colLabels=[
            "experiment",
            "success",
            "R^2",
            "redchi",
            "covar",
            "stderr",
            "bound gap",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.18)

    for column_index in range(7):
        table[(0, column_index)].set_facecolor("#eceff4")
        table[(0, column_index)].set_text_props(weight="bold")

    for row_index, row in display.iterrows():
        table_row = row_index + 1
        success_color = (
            STATUS_FILLS["green"]
            if row["success"] == "yes"
            else STATUS_FILLS["red"]
        )
        covariance_color = _diagnostic_fill_color(str(row["covariance"]))
        stderr_color = _diagnostic_fill_color(str(row["stderr"]))
        bound_color = _diagnostic_fill_color(str(row["bound_gap"]))
        table[(table_row, 1)].set_facecolor(success_color)
        table[(table_row, 4)].set_facecolor(covariance_color)
        table[(table_row, 5)].set_facecolor(stderr_color)
        table[(table_row, 6)].set_facecolor(bound_color)


def _diagnostic_fill_color(label: str) -> str:
    if label in {"ok", "yes"}:
        return STATUS_FILLS["green"]
    if label in {"missing", "at bound", "unbounded"}:
        return STATUS_FILLS["orange"]
    if label == "no":
        return STATUS_FILLS["red"]
    try:
        numeric = float(label)
    except ValueError:
        return "#f4f4f4"
    return STATUS_FILLS["green"] if numeric > 0.0 else STATUS_FILLS["orange"]


def _fit_status(
    *,
    fit: FitResult,
    covariance_missing: bool,
    stderr_missing: bool,
    parameter_at_bound: bool,
) -> str:
    if not fit.success:
        return "red"
    if covariance_missing or stderr_missing or parameter_at_bound:
        return "orange"
    return "green"


def _fit_log10_value(
    fit: FitResult,
    *,
    parameter: str,
) -> float | None:
    if not fit.success:
        return None
    model = get_model(fit.model_name)
    spec = next(
        (
            spec
            for spec in model.concentration_parameter_specs
            if spec.parameter == parameter
        ),
        None,
    )
    if spec is None or spec.fitted_parameter not in fit.parameters:
        return None
    value = float(fit.parameters[spec.fitted_parameter].value)
    if spec.fitted_scale == "log10":
        return value
    if value <= 0.0:
        return None
    return float(np.log10(value))


def _fit_parameter_stderr(
    fit: FitResult,
    *,
    parameter: str,
) -> float | str:
    if not fit.success:
        return "NA"
    spec = _fit_concentration_spec(fit, parameter=parameter)
    if spec is None:
        return "NA"
    estimate = fit.parameters.get(spec.fitted_parameter)
    if estimate is None:
        return "NA"
    if estimate.stderr is None:
        return "missing"
    return float(estimate.stderr)


def _fit_parameter_bound_gap(
    fit: FitResult,
    *,
    parameter: str,
) -> float | str:
    if not fit.success or fit.lmfit_result is None:
        return "NA"
    spec = _fit_concentration_spec(fit, parameter=parameter)
    if spec is None:
        return "NA"
    lmfit_parameter = fit.lmfit_result.params.get(spec.fitted_parameter)
    if lmfit_parameter is None:
        return "NA"

    value = float(lmfit_parameter.value)
    distances: list[float] = []
    if math.isfinite(float(lmfit_parameter.min)):
        distances.append(abs(value - float(lmfit_parameter.min)))
    if math.isfinite(float(lmfit_parameter.max)):
        distances.append(abs(float(lmfit_parameter.max) - value))
    if not distances:
        return "unbounded"
    return float(min(distances))


def _fit_concentration_spec(
    fit: FitResult,
    *,
    parameter: str,
):
    model = get_model(fit.model_name)
    return next(
        (
            spec
            for spec in model.concentration_parameter_specs
            if spec.parameter == parameter
        ),
        None,
    )


def _fit_has_missing_stderr(fit: FitResult) -> bool:
    varying = [estimate for estimate in fit.parameters.values() if estimate.vary]
    if not varying:
        return False
    return any(estimate.stderr is None for estimate in varying)


def _fit_has_parameter_at_bound(
    fit: FitResult,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if fit.lmfit_result is None:
        return False
    for parameter in fit.lmfit_result.params.values():
        if not parameter.vary:
            continue
        value = float(parameter.value)
        lower = float(parameter.min)
        upper = float(parameter.max)
        if math.isfinite(lower) and math.isclose(
            value,
            lower,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return True
        if math.isfinite(upper) and math.isclose(
            value,
            upper,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return True
    return False


def _ordered_result_compounds(results: FitResults) -> list[str]:
    compound_ids: list[str] = []
    seen: set[str] = set()
    for fit in results.fit_results:
        compound_id = str(fit.compound_id)
        if compound_id in seen:
            continue
        seen.add(compound_id)
        compound_ids.append(compound_id)
    return compound_ids


def _format_concentration_labels(concentrations: list[float]) -> list[str]:
    return [_format_concentration_label(value) for value in concentrations]


def _format_concentration_label(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.0e}"
    return f"{value:g}"


def _format_flags(flags: object) -> str:
    text = str(flags) if flags else "none"
    return text if text else "none"


def _format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(int(value))


def _format_float(value: object, places: int) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{places}f}"


def _format_diagnostic_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric == 0.0:
        return "0.00"
    if abs(numeric) < 1.0e-2 or abs(numeric) >= 1.0e3:
        return f"{numeric:.2e}"
    return f"{numeric:.3f}"


def _safe_nanmax(values: np.ndarray, *, fallback: float) -> float:
    if values.size == 0 or np.isnan(values).all():
        return fallback
    return float(np.nanmax(values))


def _relative_intensity(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 1.0
    return float((value - vmin) / (vmax - vmin))
