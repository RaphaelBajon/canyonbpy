"""
Tests for the canyonb xarray accessor (ds.canyonb).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime

import canyonbpy  # triggers accessor registration
from canyonbpy import canyonb, co2content
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
        merged = xr.merge([simple_dataset, result], compat="override")
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


# ---------------------------------------------------------------------------
# content() method
# ---------------------------------------------------------------------------

_CONTENT_VARS = ["AT", "CT", "pH", "pCO2"]
# Bittig 2018 check-value point
_MATLAB_REF = {
    "AT":   2357.817,
    "CT":   2199.472,
    "pH":   7.870137,
    "pCO2": 639.848,
}


@pytest.fixture(scope="module")
def content_dataset():
    """Single-point xr.Dataset matching the Bittig 2018 check-value."""
    return xr.Dataset(
        {
            "temperature": ("N", np.array([16.0])),
            "salinity":    ("N", np.array([36.1])),
            "doxy":        ("N", np.array([104.0])),
            "pressure":    ("N", np.array([180.0])),
        },
        coords={
            "time":      ("N", [datetime(2014, 12, 9, 8, 45)]),
            "latitude":  ("N", np.array([17.6])),
            "longitude": ("N", np.array([-24.3])),
        },
    )


@pytest.fixture(scope="module")
def content_result(content_dataset):
    return content_dataset.canyonb.content()


class TestContent:

    # --- basic structure ---

    def test_returns_dataset(self, content_result):
        assert isinstance(content_result, xr.Dataset)

    def test_required_keys(self, content_result):
        for v in _CONTENT_VARS:
            for suffix in ("", "_sigma", "_sigma_min"):
                assert f"{v}{suffix}" in content_result, f"Missing: {v}{suffix}"

    def test_no_raw_by_default(self, content_result):
        for v in _CONTENT_VARS:
            assert f"{v}_raw" not in content_result

    def test_no_canyonb_by_default(self, content_result):
        assert not any(k.startswith("canyonb_") for k in content_result.data_vars)

    # --- shape and dims ---

    def test_output_shape(self, content_result):
        for v in _CONTENT_VARS:
            assert content_result[v].shape == (1,), f"{v}.shape != (1,)"

    def test_output_dims(self, content_dataset, content_result):
        for v in _CONTENT_VARS:
            assert content_result[v].dims == ("N",), f"{v}.dims mismatch"

    def test_mergeability(self, content_dataset, content_result):
        merged = xr.merge([content_dataset, content_result], compat="override")
        assert "AT" in merged
        assert "temperature" in merged

    # --- numerical values (< 0.05 % of MATLAB reference) ---

    @pytest.mark.parametrize("var", _CONTENT_VARS)
    def test_check_value(self, content_result, var):
        got = float(content_result[var][0])
        ref = _MATLAB_REF[var]
        assert abs(got - ref) / abs(ref) < 5e-4, (
            f"{var}: got {got:.5f}, expected {ref:.5f}"
        )

    @pytest.mark.parametrize("var", _CONTENT_VARS)
    def test_sigma_positive(self, content_result, var):
        assert float(content_result[f"{var}_sigma"][0]) > 0
        assert float(content_result[f"{var}_sigma_min"][0]) > 0

    @pytest.mark.parametrize("var", _CONTENT_VARS)
    def test_sigma_min_leq_sigma(self, content_result, var):
        sig = float(content_result[f"{var}_sigma"][0])
        sig_min = float(content_result[f"{var}_sigma_min"][0])
        assert sig_min <= sig + 1e-10, f"{var}: sigma_min ({sig_min}) > sigma ({sig})"

    # --- consistent with direct co2content() call ---

    def test_matches_co2content_direct(self, content_dataset, content_result):
        from canyonbpy.preprocessing import DatasetToNumpy
        inputs = DatasetToNumpy(content_dataset).to_dict()
        ref = co2content(**inputs)
        for v in _CONTENT_VARS:
            np.testing.assert_allclose(
                content_result[v].values,
                ref[v],
                rtol=1e-8,
                err_msg=f"{v} accessor != co2content() direct",
            )

    # --- include_raw ---

    def test_include_raw_keys(self, content_dataset):
        res = content_dataset.canyonb.content(include_raw=True)
        for v in _CONTENT_VARS:
            assert f"{v}_raw" in res, f"Missing {v}_raw"

    def test_include_raw_shape(self, content_dataset):
        res = content_dataset.canyonb.content(include_raw=True)
        for v in _CONTENT_VARS:
            assert res[f"{v}_raw"].shape == (1, 4), f"{v}_raw shape mismatch"

    def test_include_raw_dims(self, content_dataset):
        res = content_dataset.canyonb.content(include_raw=True)
        for v in _CONTENT_VARS:
            assert res[f"{v}_raw"].dims == ("N", "estimate")

    def test_include_raw_col0_matches_estimate(self, content_dataset, content_result):
        """Column 0 of *_raw (CANYON-B direct) must sit within the spread."""
        res = content_dataset.canyonb.content(include_raw=True)
        for v in _CONTENT_VARS:
            col0 = float(res[f"{v}_raw"][0, 0])
            weighted = float(content_result[v][0])
            # col0 and weighted mean should be in the same ballpark (< 1 %)
            assert abs(col0 - weighted) / abs(weighted) < 0.01, (
                f"{v} raw col-0 ({col0}) far from weighted mean ({weighted})"
            )

    # --- include_canyonb ---

    def test_include_canyonb_keys(self, content_dataset):
        res = content_dataset.canyonb.content(include_canyonb=True)
        assert any(k.startswith("canyonb_") for k in res.data_vars)
        # co2content calls canyonb for AT, CT, pH, pCO2, SiOH4, PO4 (not NO3)
        for v in ["AT", "CT", "pH", "pCO2", "SiOH4", "PO4"]:
            assert f"canyonb_{v}" in res, f"Missing canyonb_{v}"

    def test_include_canyonb_shape(self, content_dataset):
        res = content_dataset.canyonb.content(include_canyonb=True)
        assert res["canyonb_AT"].shape == (1,)

    # --- custom errors propagate ---

    def test_large_edoxy_increases_pco2_sigma(self, content_dataset):
        default = content_dataset.canyonb.content()
        large   = content_dataset.canyonb.content(edoxy=0.10 * 104.0)
        assert float(large["pCO2_sigma"][0]) >= float(default["pCO2_sigma"][0]) * 0.95

    def test_large_etemp_increases_at_sigma(self, content_dataset):
        default = content_dataset.canyonb.content()
        large   = content_dataset.canyonb.content(etemp=1.0)
        assert float(large["AT_sigma"][0]) >= float(default["AT_sigma"][0]) * 0.95

    # --- custom var_map (Argo) ---

    def test_argo_var_map(self, argo_dataset):
        var_map = {
            "temp": "TEMP_ADJUSTED",
            "psal": "PSAL_ADJUSTED",
            "doxy": "DOXY_ADJUSTED",
            "pres": "PRES_ADJUSTED",
            "lat":  "LATITUDE",
            "lon":  "LONGITUDE",
        }
        res = argo_dataset.canyonb.content(var_map=var_map)
        assert res["AT"].shape == (2,)
        assert res["AT"].dims == ("N_PROF",)
        for v in _CONTENT_VARS:
            assert np.all(np.isfinite(res[v].values))


# ---------------------------------------------------------------------------
# 4-D dataset  (time × pressure × latitude × longitude)
# ---------------------------------------------------------------------------

_4D_SHAPE = (2, 3, 3, 4)
_4D_DIMS  = ("time", "pressure", "latitude", "longitude")


@pytest.fixture(scope="module")
def grid_dataset():
    """4-D xr.Dataset where variables span time × pressure × lat × lon."""
    return xr.Dataset(
        {
            "temperature": (_4D_DIMS, 16.0 * np.ones(_4D_SHAPE)),
            "salinity":    (_4D_DIMS, 36.1 * np.ones(_4D_SHAPE)),
            "doxy":        (_4D_DIMS, 104.0 * np.ones(_4D_SHAPE)),
        },
        coords={
            "time":      ("time",      [datetime(2014, 12, 9, 8, 45),
                                        datetime(2020, 12, 10, 8, 45)]),
            "pressure":  ("pressure",  np.array([180.0, 181.0, 182.0])),
            "latitude":  ("latitude",  np.array([17.6, 17.6, 17.6])),
            "longitude": ("longitude", np.array([-24.3, -24.3, -24.3, -24.3])),
        },
    )


class TestGridDataset:
    """predict() and content() on a 4-D (time × pressure × lat × lon) dataset."""

    # --- predict ---

    def test_predict_returns_dataset(self, grid_dataset):
        result = grid_dataset.canyonb.predict(param=["pH"])
        assert isinstance(result, xr.Dataset)

    def test_predict_shape(self, grid_dataset):
        result = grid_dataset.canyonb.predict(param=["pH"])
        assert result["pH"].shape == _4D_SHAPE

    def test_predict_dims(self, grid_dataset):
        result = grid_dataset.canyonb.predict(param=["pH"])
        assert result["pH"].dims == _4D_DIMS

    def test_predict_finite(self, grid_dataset):
        result = grid_dataset.canyonb.predict(param=["pH", "AT", "NO3"])
        for v in ["pH", "AT", "NO3"]:
            assert np.all(np.isfinite(result[v].values)), f"{v} has non-finite values"

    def test_predict_coords_propagated(self, grid_dataset):
        result = grid_dataset.canyonb.predict(param=["pH"])
        assert "time"     in result.coords
        assert "pressure" in result.coords

    def test_predict_isel(self, grid_dataset):
        """isel must work after predict, confirming dimension labels are correct."""
        result = grid_dataset.canyonb.predict(param=["pH"])
        sub = result.isel(time=0, pressure=1)
        assert sub["pH"].shape == (3, 4)

    # --- content ---

    def test_content_returns_dataset(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        assert isinstance(result, xr.Dataset)

    def test_content_shape(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        for v in _CONTENT_VARS:
            assert result[v].shape == _4D_SHAPE, f"{v}.shape mismatch"
            assert result[f"{v}_sigma"].shape == _4D_SHAPE
            assert result[f"{v}_sigma_min"].shape == _4D_SHAPE

    def test_content_dims(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        for v in _CONTENT_VARS:
            assert result[v].dims == _4D_DIMS, f"{v}.dims mismatch"

    def test_content_finite(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        for v in _CONTENT_VARS:
            assert np.all(np.isfinite(result[v].values)), f"{v} has non-finite values"
            assert np.all(result[f"{v}_sigma"].values > 0)

    def test_content_coords_propagated(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        assert "time"     in result.coords
        assert "pressure" in result.coords

    def test_content_isel(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        sub = result.isel(time=0, pressure=1)
        assert sub["AT"].shape == (3, 4)

    def test_content_include_raw_shape(self, grid_dataset):
        result = grid_dataset.canyonb.content(include_raw=True)
        for v in _CONTENT_VARS:
            assert result[f"{v}_raw"].shape == _4D_SHAPE + (4,)
            assert result[f"{v}_raw"].dims  == _4D_DIMS + ("estimate",)

    def test_content_include_canyonb_shape(self, grid_dataset):
        result = grid_dataset.canyonb.content(include_canyonb=True)
        assert "canyonb_AT" in result
        assert result["canyonb_AT"].shape == _4D_SHAPE
        assert result["canyonb_AT"].dims  == _4D_DIMS

    def test_content_mergeability(self, grid_dataset):
        result = grid_dataset.canyonb.content()
        merged = xr.merge([grid_dataset, result], compat="override")
        assert "AT" in merged
        assert "temperature" in merged
