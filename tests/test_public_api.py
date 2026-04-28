from __future__ import annotations

import importlib

import bindcurve as bc


def test_public_api_exports_new_objects():
    expected = {
        "BaseDoseResponseModel",
        "CompoundData",
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
        "IC50Model",
        "LogIC50Model",
        "ParameterEstimate",
        "ParameterSpec",
        "fit",
        "get_model",
    }

    assert expected <= set(bc.__all__)


def test_legacy_flat_modules_are_not_importable():
    for module_name in ["bindcurve.data", "bindcurve.calculate", "bindcurve.models"]:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Legacy module {module_name!r} should not be importable")
