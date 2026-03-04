"""
Tests for the canyonb xarray accessor (ds.canyonb).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime

import canyonbpy  # noqa: F401 — triggers accessor registration
from canyonbpy import canyonb
from canyonbpy.preprocessing import DatasetToNumpy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dataset():
    """1-D xr.Dataset using default variable names."""
    n = 3
    times = pd.date_range("2024-01-01", periods=n, freq="D")
    return xr.Dataset(
        {
            "temperature": ("points", np.array([15.0, 16.0, 14.0])),
            "salinity":    ("points", np.array([35.0, 35.1, 34.9])),
            "doxy":        ("points", np.array([250.0, 245.0, 255.0])),
            "pressure":    ("points", np.array([100.0, 200.0, 300.0])),
        },
        coords={
            "time":      ("points", times),
            "latitude":  ("points", np.array([45.0, 46.0, 47.0])),
            "longitude": ("points", np.array([-20.0, -21.0, -22.0])),
        },
    )


@pytest.fixture
def argo_dataset():
    """Dataset with Argo BGC delayed-mode variable names."""
    n = 2
    times = pd.date_range("2024-06-01", periods=n, freq="D")
    return xr.Dataset(
        {
            "TEMP_ADJUSTED": ("N_PROF", np.array([12.0, 13.0])),
            "PSAL_ADJUSTED": ("N_PROF", np.array([35.2, 35.3])),
            "DOXY_ADJUSTED": ("N_PROF", np.array([230.0, 235.0])),
            "PRES_ADJUSTED": ("N_PROF", np.array([500.0, 600.0])),
        },
        coords={
            "time":     ("N_PROF", times),
            "LATITUDE": ("N_PROF", np.array([-50.0, -51.0])),
            "LONGITUDE":("N_PROF", np.array([10.0, 11.0])),
        },
    )


# ---------------------------------------------------------------------------
# Accessor registration
# ---------------------------------------------------------------------------

class TestAccessorRegistration:

    def test_accessor_is_available(self, simple_dataset):
        assert hasattr(simple_dataset, "canyonb")

    def test_accessor_type(self, simple_dataset):
        from canyonbpy.accessor import CanyonBAccessor
        assert isinstance(simple_dataset.canyonb, CanyonBAccessor)


# ---------------------------------------------------------------------------
# predict() method
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_dataset(self, simple_dataset):
        result = simple_dataset.canyonb.predict(param=["pH"])
        assert isinstance(result, xr.Dataset)

    def test_expected_variables_present(self, simple_dataset):
        result = simple_dataset.canyonb.predict(param=["pH", "AT"])
        assert "pH" in result
        assert "pH_ci" in result
        assert "AT" in result
        assert "AT_ci" in result

    def test_unrequested_params_absent(self, simple_dataset):
        result = simple_dataset.canyonb.predict(param=["pH"])
        assert "NO3" not in result

    def test_output_shape_preserved(self, simple_dataset):
        result = simple_dataset.canyonb.predict(param=["NO3"])
        assert result["NO3"].shape == (3,)

    def test_output_dims_match_input(self, simple_dataset):
        result = simple_dataset.canyonb.predict(param=["pH"])
        assert result["pH"].dims == ("points",)

    def test_result_is_mergeable(self, simple_dataset):
        result = simple_dataset.canyonb.predict(param=["pH"])
        merged = xr.merge([simple_dataset, result])
        assert "pH" in merged
        assert "temperature" in merged

    def test_custom_var_map_argo(self, argo_dataset):
        var_map = {
            "temp": "TEMP_ADJUSTED",
            "psal": "PSAL_ADJUSTED",
            "doxy": "DOXY_ADJUSTED",
            "pres": "PRES_ADJUSTED",
            "lat":  "LATITUDE",
            "lon":  "LONGITUDE",
        }
        result = argo_dataset.canyonb.predict(var_map=var_map, param=["pH"])
        assert "pH" in result
        assert result["pH"].shape == (2,)
        assert result["pH"].dims == ("N_PROF",)

    def test_results_consistent_with_canyonb(self, simple_dataset):
        """Accessor and canyonb() must give identical numerical results."""
        conv = DatasetToNumpy(simple_dataset)
        ref = canyonb(**conv.to_dict(), param=["pH"])

        result = simple_dataset.canyonb.predict(param=["pH"])
        np.testing.assert_array_almost_equal(ref["pH"], result["pH"].values)

    def test_custom_errors(self, simple_dataset):
        result = simple_dataset.canyonb.predict(
            param=["pH"], epres=1.0, etemp=0.01, epsal=0.01, edoxy=0.02
        )
        assert "pH" in result


# ---------------------------------------------------------------------------
# converter() method
# ---------------------------------------------------------------------------

class TestConverter:

    def test_returns_dataset_to_numpy(self, simple_dataset):
        conv = simple_dataset.canyonb.converter()
        assert isinstance(conv, DatasetToNumpy)

    def test_converter_with_var_map(self, argo_dataset):
        var_map = {
            "temp": "TEMP_ADJUSTED",
            "psal": "PSAL_ADJUSTED",
            "doxy": "DOXY_ADJUSTED",
            "pres": "PRES_ADJUSTED",
            "lat":  "LATITUDE",
            "lon":  "LONGITUDE",
        }
        conv = argo_dataset.canyonb.converter(var_map=var_map)
        inputs = conv.to_dict()
        np.testing.assert_array_equal(inputs["temp"], np.array([12.0, 13.0]))
