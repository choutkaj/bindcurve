from __future__ import annotations

import importlib

import bindcurve as bc


def test_public_api_exports_new_objects():
    expected = {
        "BaseDoseResponseModel",
        "ConcentrationSummary",
        "CurvePoint",
        "DataQualityThresholds",
        "DoseResponseData",
        "FitMetrics",
        "FitResult",
        "FitResults",
        "FitSettings",
        "IC50ConversionResult",
        "ModelEvaluation",
        "ParameterEstimate",
        "ParameterSummary",
        "ParameterSpec",
        "ResultQualityThresholds",
        "cheng_prusoff",
        "cheng_prusoff_corrected",
        "coleska",
        "convert_ic50_to_kd",
        "fit",
        "get_model",
        "plot_asymptotes",
        "plot_compounds",
        "plot_curve_points",
        "plot_fits",
        "plot_residuals",
    }

    assert set(bc.__all__) == expected | {"__version__"}


def test_low_level_implementation_objects_are_not_exported_at_package_root():
    removed = {
        "CompoundData",
        "FitCalculator",
        "IC50Model",
        "plot_confidence_bands",
        "plot_fit_lines",
        "plot_observations",
    }

    assert removed.isdisjoint(bc.__all__)


def test_legacy_flat_modules_are_not_importable():
    for module_name in ["bindcurve.data", "bindcurve.calculate", "bindcurve.models"]:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Legacy module {module_name!r} should not be importable")
