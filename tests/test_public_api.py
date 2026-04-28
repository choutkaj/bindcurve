from __future__ import annotations

import importlib

import bindcurve as bc


def test_public_api_exports_new_objects():
    expected = {
        "BaseDoseResponseModel",
        "CompetitiveFourStateSpecificKdModel",
        "CompetitiveFourStateTotalKdModel",
        "CompetitiveThreeStateSpecificKdModel",
        "CompetitiveThreeStateTotalKdModel",
        "CompoundData",
        "CurvePoint",
        "DirectSimpleKdModel",
        "DirectSpecificKdModel",
        "DirectTotalKdModel",
        "DoseResponseData",
        "EC50Model",
        "FitCalculator",
        "FitMetrics",
        "FitResult",
        "FitResults",
        "FitSettings",
        "IC50ConversionResult",
        "IC50Model",
        "LogIC50Model",
        "ParameterEstimate",
        "ParameterSpec",
        "cheng_prusoff",
        "cheng_prusoff_corrected",
        "coleska",
        "convert_ic50_to_kd",
        "fit",
        "get_model",
        "plot_asymptotes",
        "plot_confidence_bands",
        "plot_curve_points",
        "plot_curves",
        "plot_fits",
        "plot_observations",
        "plot_residuals",
    }

    assert expected <= set(bc.__all__)


def test_legacy_flat_modules_are_not_importable():
    for module_name in ["bindcurve.data", "bindcurve.calculate", "bindcurve.models"]:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Legacy module {module_name!r} should not be importable")
