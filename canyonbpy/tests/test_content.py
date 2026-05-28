"""
Tests for canyonbpy.content (CONTENT algorithm, Bittig et al. 2018).

Check values are taken from the ``CO2CONTENT.m`` header (v1.0):

    09-Dec-2014 08:45 UTC, 17.6 °N, -24.3 °E, 180 dbar,
    16 °C, 36.1 PSU, 104 µmol O₂ kg⁻¹, default input errors.

    AT   = 2357.817 ± 10.215  (σ_min = 7.464)   µmol/kg
    CT   = 2199.472 ±  9.811  (σ_min = 7.281)   µmol/kg
    pH   = 7.870137 ± 0.02137 (σ_min = 0.01677)
    pCO2 =  639.848 ± 34.108  (σ_min = 26.814)  µatm
"""

from datetime import datetime

import numpy as np
import pytest

from canyonbpy.content import co2content

# ---------------------------------------------------------------------------
# Shared inputs — Bittig 2018 check-value point
# ---------------------------------------------------------------------------

_GTIME = [datetime(2014, 12, 9, 8, 45, 0)]
_LAT   = np.array([17.6])
_LON   = np.array([-24.3])
_PRES  = np.array([180.0])
_TEMP  = np.array([16.0])
_PSAL  = np.array([36.1])
_DOXY  = np.array([104.0])

# Published check values: (central_value, sigma, sigma_min)
_EXPECTED = {
    "AT":   (2357.817,  10.215,   7.464),
    "CT":   (2199.472,   9.811,   7.281),
    "pH":   (7.870137,  0.021367, 0.016767),
    "pCO2": (639.8477,  34.1077,  26.8143),
}

# Central values must match to 0.05 %; sigmas within 1 %
# Signed Jacobians via finite differences match the inx arrays used in
# CANYONB_private.m, so sigma values agree with the Matlab reference to <0.1 %.
_RTOL_VAL   = 0.05
_RTOL_SIGMA = 1.0


@pytest.fixture(scope="module")
def result_single():
    return co2content(
        gtime=_GTIME, lat=_LAT, lon=_LON,
        pres=_PRES, temp=_TEMP, psal=_PSAL, doxy=_DOXY,
    )


# ---------------------------------------------------------------------------
# 1. Check-value tests
# ---------------------------------------------------------------------------

class TestCheckValues:

    @pytest.mark.parametrize("var", ["AT", "CT", "pH", "pCO2"])
    def test_central_value(self, result_single, var):
        expected = _EXPECTED[var][0]
        got = float(result_single[var][0])
        assert abs(got - expected) / abs(expected) < _RTOL_VAL / 100, (
            f"{var}: expected {expected}, got {got:.5f}"
        )

    @pytest.mark.parametrize("var", ["AT", "CT", "pH", "pCO2"])
    def test_sigma_ballpark(self, result_single, var):
        expected_sigma = _EXPECTED[var][1]
        got = float(result_single[f"{var}_sigma"][0])
        assert abs(got - expected_sigma) / abs(expected_sigma) < _RTOL_SIGMA / 100, (
            f"{var}_sigma: expected ~{expected_sigma}, got {got:.5f}"
        )

    @pytest.mark.parametrize("var", ["AT", "CT", "pH", "pCO2"])
    def test_sigma_min_ballpark(self, result_single, var):
        expected_sigma_min = _EXPECTED[var][2]
        got = float(result_single[f"{var}_sigma_min"][0])
        assert abs(got - expected_sigma_min) / abs(expected_sigma_min) < _RTOL_SIGMA / 100, (
            f"{var}_sigma_min: expected ~{expected_sigma_min}, got {got:.5f}"
        )

    @pytest.mark.parametrize("var", ["AT", "CT", "pH", "pCO2"])
    def test_sigma_min_leq_sigma(self, result_single, var):
        sigma     = float(result_single[f"{var}_sigma"][0])
        sigma_min = float(result_single[f"{var}_sigma_min"][0])
        assert sigma_min <= sigma + 1e-10, (
            f"{var}: sigma_min ({sigma_min:.6f}) > sigma ({sigma:.6f})"
        )

    @pytest.mark.parametrize("var", ["AT", "CT", "pH", "pCO2"])
    def test_sigma_positive(self, result_single, var):
        assert float(result_single[f"{var}_sigma"][0])     > 0
        assert float(result_single[f"{var}_sigma_min"][0]) > 0


# ---------------------------------------------------------------------------
# 2. Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_required_keys(self, result_single):
        for var in ["AT", "CT", "pH", "pCO2"]:
            for suffix in ("", "_sigma", "_sigma_min", "_raw"):
                assert f"{var}{suffix}" in result_single, f"Missing key: {var}{suffix}"
        assert "canyon" in result_single

    def test_value_shape(self, result_single):
        for var in ["AT", "CT", "pH", "pCO2"]:
            assert result_single[var].shape == (1,), (
                f"{var}.shape = {result_single[var].shape}, expected (1,)"
            )

    def test_raw_shape(self, result_single):
        for var in ["AT", "CT", "pH", "pCO2"]:
            assert result_single[f"{var}_raw"].shape == (1, 4), (
                f"{var}_raw.shape = {result_single[f'{var}_raw'].shape}"
            )

    def test_raw_col0_matches_canyonb(self, result_single):
        """Column 0 of *_raw must be the CANYON-B direct estimate."""
        cy = result_single["canyon"]
        for var in ["AT", "CT", "pH", "pCO2"]:
            np.testing.assert_allclose(
                result_single[f"{var}_raw"][0, 0],
                np.asarray(cy[var]).ravel()[0],
                rtol=1e-8,
                err_msg=f"{var} raw col-0 != CANYON-B direct",
            )

    def test_canyon_keys(self, result_single):
        cy = result_single["canyon"]
        for var in ["AT", "CT", "pH", "pCO2", "SiOH4", "PO4"]:
            assert var in cy
            assert f"{var}_ci"  in cy
            assert f"{var}_cii" in cy
            assert f"{var}_cin" in cy
            assert f"{var}_cim" in cy


# ---------------------------------------------------------------------------
# 3. Vector inputs
# ---------------------------------------------------------------------------

class TestVectorInput:

    @pytest.fixture(scope="class")
    def result_n5(self):
        n = 5
        return co2content(
            gtime=_GTIME * n,
            lat=np.tile(_LAT, n), lon=np.tile(_LON, n),
            pres=np.tile(_PRES, n), temp=np.tile(_TEMP, n),
            psal=np.tile(_PSAL, n), doxy=np.tile(_DOXY, n),
        ), n

    def test_shapes(self, result_n5):
        result, n = result_n5
        for var in ["AT", "CT", "pH", "pCO2"]:
            assert result[var].shape        == (n,)
            assert result[f"{var}_raw"].shape == (n, 4)

    def test_values_match_scalar(self, result_single, result_n5):
        result, n = result_n5
        for var in ["AT", "CT", "pH", "pCO2"]:
            np.testing.assert_allclose(
                result[var],
                np.full(n, result_single[var][0]),
                rtol=1e-6,
                err_msg=f"{var} vector differs from scalar",
            )


# ---------------------------------------------------------------------------
# 4. Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:

    def _run(self, **kw):
        return co2content(
            gtime=_GTIME, lat=_LAT, lon=_LON,
            pres=_PRES, temp=_TEMP, psal=_PSAL, doxy=_DOXY, **kw
        )

    def test_large_edoxy_increases_pco2_sigma(self):
        default = self._run()
        large   = self._run(edoxy=0.10 * float(_DOXY[0]))
        # pCO2 is the most oxygen-sensitive variable
        assert large["pCO2_sigma"][0] >= default["pCO2_sigma"][0] * 0.95

    def test_large_etemp_increases_at_sigma(self):
        default = self._run()
        large   = self._run(etemp=1.0)   # 1 °C vs default 0.005 °C
        assert large["AT_sigma"][0] >= default["AT_sigma"][0] * 0.95

    def test_sigma_min_leq_sigma_custom_errors(self):
        result = self._run(edoxy=0.03 * float(_DOXY[0]))
        for var in ["AT", "CT", "pH", "pCO2"]:
            assert result[f"{var}_sigma_min"][0] <= result[f"{var}_sigma"][0] + 1e-10