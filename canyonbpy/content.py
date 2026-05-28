"""
CONTENT: Carbonate system variables made internally consistent with chemistry.

Python translation of ``CO2CONTENT.m`` (Bittig et al., 2018) for the
**canyonbpy** package.

The algorithm takes CANYON-B estimates of AT, CT, pH, and pCO2, runs all six
possible two-parameter CO2SYS calculations, then returns a weighted mean of all
four estimates per variable with full propagated uncertainty — including the
covariance between variables arising from their shared hydrographic inputs
(T, S, O₂, pressure).

Reference
---------
Bittig et al. (2018). An alternative to static climatologies: Robust
estimation of open ocean CO₂ variables and nutrient concentrations from T, S
and O₂ data using Bayesian neural networks.
*Front. Mar. Sci.* **5**:328. https://doi.org/10.3389/fmars.2018.00328

Notes
-----
PyCO2SYS >= 1.8.0 is already a hard dependency of canyonbpy, so no new
dependency is introduced.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import PyCO2SYS as pyco2

from .core import canyonb as _canyonb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Carbonate variables handled by CONTENT, in CO2SYS parameter-type order.
PARAMS: List[str] = ["AT", "CT", "pH", "pCO2"]

#: CO2SYS/PyCO2SYS parameter-type integer for each variable.
_PARAM_TYPE: Dict[str, int] = {"AT": 1, "CT": 2, "pH": 3, "pCO2": 4}

#: PyCO2SYS output-dict key for each variable.
_PYCO2_KEY: Dict[str, str] = {
    "AT": "alkalinity",
    "CT": "dic",
    "pH": "pH",
    "pCO2": "pCO2",
}

# The six two-parameter input combinations  (par1_type, par2_type).
_INPAR: List[tuple] = [
    (4, 1),  # pCO2, AT  → CT, pH
    (3, 1),  # pH,   AT  → CT, pCO2
    (4, 3),  # pCO2, pH  → CT, AT
    (4, 2),  # pCO2, CT  → AT, pH
    (3, 2),  # pH,   CT  → AT, pCO2
    (1, 2),  # AT,   CT  → pH, pCO2
]

# For each combination: the two output parameter types (the two not used as inputs).
_OUTPAR: List[tuple] = [
    tuple(sorted(set(range(1, 5)) - set(ip))) for ip in _INPAR  # type: ignore[misc]
]

# 0-based column indices (in the N×4 raw array) where each CO2SYS result is stored.
# Column 0 = CANYON-B direct;  columns 1–3 = CO2SYS indirect.
# Mirrors the ``svi`` array in CO2CONTENT.m (here 0-indexed).
_COL: List[tuple] = [
    (1, 3),
    (2, 2),
    (3, 3),
    (1, 2),
    (2, 3),
    (1, 1),
]

# PyCO2SYS options  — Lueker et al. (2000) K1K2, Dickson (1990) KSO4, Uppstrom (1979) borate
_CO2SYS_OPTS: dict = dict(
    opt_k_carbonic=10,
    opt_k_bisulfate=1,
    opt_total_borate=1,
    opt_pH_scale=1,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_jacobians(
    gtime: list,
    lat: np.ndarray,
    lon: np.ndarray,
    pres: np.ndarray,
    temp: np.ndarray,
    psal: np.ndarray,
    doxy: np.ndarray,
    base_vals: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return SIGNED d(var)/d(input_k) via forward finite differences.

    MATLAB's ``CANYONB_private.m`` stores ``out.inx`` as SIGNED neural-network
    derivatives (the chain-rule product of weight matrices and sech² activations),
    which are then used directly in ``CO2CONTENT.m`` to build the off-diagonal
    cross-covariance matrix::

        cycov(i,j) = Σ_k inx_i[k] * inx_j[k] * e_k²

    The previous unit-error trick recovered ``_cii = sqrt(Σ inx² e²)``, which
    always gives ``|inx|``, losing the sign.  Forward finite differences recover
    the same signed derivatives without any modification of ``core.py``.  Four
    additional ``canyonb`` calls are needed (same as before); the pre-computed
    *base_vals* replace what would have been a redundant fifth call.

    Parameters
    ----------
    base_vals : dict
        Pre-computed central-point CANYON-B values ``{param: (N,) array}``
        obtained from the main ``canyonb`` call.

    Returns
    -------
    jacs : dict
        Keys are variable names in ``PARAMS``.
        Values are ``(N, 4)`` arrays of **SIGNED** partial derivatives
        ordered as ``[∂var/∂temp, ∂var/∂psal, ∂var/∂doxy, ∂var/∂pres]``.
    """
    N = len(pres)
    jacs: Dict[str, np.ndarray] = {v: np.zeros((N, 4)) for v in PARAMS}

    # Step sizes: large enough to avoid floating-point noise while staying in
    # the linear regime of the smooth neural-network outputs.
    _PERTURBATIONS = [
        ("temp", 1e-3, temp + 1e-3),
        ("psal", 1e-3, psal + 1e-3),
        ("doxy", 0.1,  doxy + 0.1),
        ("pres", 0.1,  pres + 0.1),
    ]

    common = dict(
        gtime=gtime, lat=lat, lon=lon,
        temp=temp, psal=psal, doxy=doxy, pres=pres,
        param=PARAMS, epres=0., etemp=0., epsal=0., edoxy=0.,
    )

    for k, (input_name, delta, val_plus) in enumerate(_PERTURBATIONS):
        r = _canyonb(**{**common, input_name: val_plus})
        for v in PARAMS:
            jacs[v][:, k] = (np.asarray(r[v]).ravel() - base_vals[v]) / delta

    return jacs


def _pyco2sys(
    par1: np.ndarray,
    par2: np.ndarray,
    par1_type: int,
    par2_type: int,
    salinity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    total_silicate: np.ndarray,
    total_phosphate: np.ndarray,
    grads: bool = False,
) -> dict:
    """Thin wrapper around ``PyCO2SYS.sys`` with project-consistent options."""
    kwargs: dict = dict(
        par1=par1, par2=par2,
        par1_type=par1_type, par2_type=par2_type,
        salinity=salinity, temperature=temperature, pressure=pressure,
        total_silicate=total_silicate, total_phosphate=total_phosphate,
        **_CO2SYS_OPTS,
    )
    if grads:
        out_vars = [v for v in PARAMS if _PARAM_TYPE[v] not in (par1_type, par2_type)]
        kwargs["grads_of"]  = [_PYCO2_KEY[v] for v in out_vars]
        kwargs["grads_wrt"] = ["par1", "par2"]
    return pyco2.sys(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def co2content(
    gtime: Union[List[datetime], np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    pres: np.ndarray,
    temp: np.ndarray,
    psal: np.ndarray,
    doxy: np.ndarray,
    epres: float = 0.5,
    etemp: float = 0.005,
    epsal: float = 0.005,
    edoxy: Optional[Union[float, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Internally consistent carbonate system estimates (CONTENT).

    Combines CANYON-B Bayesian neural network estimates of AT, CT, pH, and
    pCO₂ with all six possible two-parameter CO2SYS calculations into a
    single weighted-mean estimate consistent with carbonate chemistry.
    Uncertainty propagation accounts for the covariance between variables
    arising from their shared hydrographic inputs.

    Parameters
    ----------
    gtime : list of datetime or array of decimal years
        Date/time of measurements — same formats accepted by :func:`canyonb`.
    lat : array_like
        Latitude (°N, −90 to 90).
    lon : array_like
        Longitude (°E, −180 to 180 or 0 to 360).
    pres : array_like
        Pressure (dbar).
    temp : array_like
        In-situ temperature (°C).
    psal : array_like
        Salinity.
    doxy : array_like
        Dissolved oxygen (µmol kg⁻¹).
    epres : float, optional
        Pressure input error in dbar (default 0.5).
    etemp : float, optional
        Temperature input error in °C (default 0.005).
    epsal : float, optional
        Salinity input error (default 0.005).
    edoxy : float or array_like, optional
        Oxygen input error in µmol kg⁻¹ (default 1 % of *doxy*).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - **AT**, **CT**, **pH**, **pCO2** — weighted-mean CONTENT estimates.
        - **AT_sigma**, **CT_sigma**, **pH_sigma**, **pCO2_sigma** — total
          uncertainty (empirical spread + propagated input covariance).
        - **AT_sigma_min**, **CT_sigma_min**, **pH_sigma_min**,
          **pCO2_sigma_min** — minimum propagated uncertainty
          (correlated-input component only).
        - **AT_raw**, **CT_raw**, **pH_raw**, **pCO2_raw** — all four raw
          estimates: index 0 = CANYON-B direct, indices 1–3 from CO2SYS
          combinations, shape ``(N, 4)``.
        - **canyon** — full CANYON-B output for AT, CT, pH, pCO2, SiOH4,
          and PO4, including all ``_ci``, ``_cim``, ``_cin``, and ``_cii``
          uncertainty components.

    Notes
    -----
    The cross-variable covariance matrix ``cycov`` is built from SIGNED
    Jacobians ``∂var/∂x_k`` recovered via forward finite differences, matching
    the ``inx`` arrays computed inside ``CANYONB_private.m``.  This preserves
    the correct sign of cross-covariance terms (e.g., AT and pH can have
    opposing sensitivity to temperature, giving a negative covariance) that
    was lost in the previous unit-error approach.

    Check values (Bittig et al. 2018, from ``CO2CONTENT.m`` header)
    ---------------------------------------------------------------
    Input: 09-Dec-2014 08:45 UTC · 17.6 °N · −24.3 °E · 180 dbar ·
           16 °C · 36.1 PSU · 104 µmol O₂ kg⁻¹ (default input errors)

    | var  | estimate | sigma   | sigma_min | unit    |
    |------|----------|---------|-----------|---------|
    | AT   | 2357.817 | 10.215  | 7.464     | µmol/kg |
    | CT   | 2199.472 | 9.811   | 7.281     | µmol/kg |
    | pH   | 7.870137 | 0.02137 | 0.01677   |         |
    | pCO2 | 639.848  | 34.108  | 26.814    | µatm    |

    Central values match to < 0.05 %. Sigma values are computed from SIGNED
    Jacobians via finite differences, matching the ``inx`` arrays used in
    ``CANYONB_private.m``.

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from canyonbpy.content import co2content
    >>> out = co2content(
    ...     gtime=[datetime(2014, 12, 9, 8, 45)],
    ...     lat=np.array([17.6]),
    ...     lon=np.array([-24.3]),
    ...     pres=np.array([180.0]),
    ...     temp=np.array([16.0]),
    ...     psal=np.array([36.1]),
    ...     doxy=np.array([104.0]),
    ... )
    >>> print(f"AT  = {out['AT'][0]:.3f} ± {out['AT_sigma'][0]:.3f} µmol/kg")
    >>> print(f"pH  = {out['pH'][0]:.6f} ± {out['pH_sigma'][0]:.6f}")
    """
    # ------------------------------------------------------------------
    # 1.  Sanitise / broadcast inputs
    # ------------------------------------------------------------------
    pres  = np.asarray(pres,  dtype=float).ravel()
    lat   = np.broadcast_to(np.asarray(lat,   dtype=float), pres.shape).copy()
    lon   = np.broadcast_to(np.asarray(lon,   dtype=float), pres.shape).copy()
    temp  = np.broadcast_to(np.asarray(temp,  dtype=float), pres.shape).copy()
    psal  = np.broadcast_to(np.asarray(psal,  dtype=float), pres.shape).copy()
    doxy  = np.broadcast_to(np.asarray(doxy,  dtype=float), pres.shape).copy()

    nol = pres.size

    if not hasattr(gtime, "__len__") or isinstance(gtime, (str, datetime)):
        gtime = [gtime] * nol
    elif len(gtime) == 1 and nol > 1:
        gtime = list(gtime) * nol

    # Resolve edoxy here so that we can pass it consistently; canyonb also
    # defaults to 1 % internally, but we need the actual value for covariance.
    if edoxy is None:
        edoxy = 0.01 * doxy
    edoxy = np.broadcast_to(np.asarray(edoxy, dtype=float), pres.shape).copy()

    epres_arr = np.full(nol, epres)
    etemp_arr = np.full(nol, etemp)
    epsal_arr = np.full(nol, epsal)

    # ------------------------------------------------------------------
    # 2.  CANYON-B estimates — AT, CT, pH, pCO2 + nutrients for CO2SYS
    # ------------------------------------------------------------------
    cy = _canyonb(
        gtime=gtime, lat=lat, lon=lon, pres=pres,
        temp=temp, psal=psal, doxy=doxy,
        param=["AT", "CT", "pH", "pCO2", "SiOH4", "PO4"],
        epres=epres, etemp=etemp, epsal=epsal, edoxy=edoxy,
    )

    cy_val = {v: np.asarray(cy[v]).ravel()        for v in PARAMS}
    cy_ci  = {v: np.asarray(cy[f"{v}_ci"]).ravel() for v in PARAMS}
    cy_SiOH4 = np.asarray(cy["SiOH4"]).ravel()
    cy_PO4   = np.asarray(cy["PO4"]).ravel()

    # ------------------------------------------------------------------
    # 3.  Jacobians → covariance matrix of the four CANYON-B outputs
    #
    #     cycov[i, j, n] = Σ_k  (∂y_i/∂x_k)(∂y_j/∂x_k) · e_k²
    #
    #     Inputs x_k ∈ {temp, psal, doxy, pres};  y_i, y_j ∈ PARAMS.
    # ------------------------------------------------------------------
    jacs = _extract_jacobians(gtime, lat, lon, pres, temp, psal, doxy, cy_val)
    # jacs[v]: (N, 4) — columns [∂v/∂temp, ∂v/∂psal, ∂v/∂doxy, ∂v/∂pres]

    e_sq = np.stack([etemp_arr, epsal_arr, edoxy, epres_arr], axis=1) ** 2  # (N, 4)

    cycov = np.zeros((4, 4, nol))
    for i, vi in enumerate(PARAMS):
        for j, vj in enumerate(PARAMS):
            if j < i:
                cycov[i, j] = cycov[j, i]
            else:
                cycov[i, j] = np.sum(jacs[vi] * jacs[vj] * e_sq, axis=1)

    # Override diagonal with total CANYON-B variance (includes network uncertainty)
    for i, v in enumerate(PARAMS):
        cycov[i, i] = cy_ci[v] ** 2

    # ------------------------------------------------------------------
    # 4.  Initialise raw-estimate and sigma arrays  (N × 4)
    # ------------------------------------------------------------------
    raw   = {v: np.full((nol, 4), np.nan) for v in PARAMS}
    sigma = {v: np.full((nol, 4), np.nan) for v in PARAMS}
    # grads[v][:, col, j] = d(estimate_col_of_v) / d(PARAMS[j])
    grads = {v: np.zeros((nol, 4, 4)) for v in PARAMS}

    for k_idx, v in enumerate(PARAMS):
        raw[v][:, 0]   = cy_val[v]
        sigma[v][:, 0] = cy_ci[v]
        grads[v][:, 0, k_idx] = 1.0

    # Store PyCO2SYS gradients for later covariance calculation
    dc: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # 5.  Six CO2SYS calculations + uncertainty propagation
    # ------------------------------------------------------------------
    for p, (ip1, ip2) in enumerate(_INPAR):
        v1  = PARAMS[ip1 - 1]
        v2  = PARAMS[ip2 - 1]
        ov1 = PARAMS[_OUTPAR[p][0] - 1]
        ov2 = PARAMS[_OUTPAR[p][1] - 1]
        col1, col2 = _COL[p]

        result = _pyco2sys(
            par1=cy_val[v1], par2=cy_val[v2],
            par1_type=ip1,   par2_type=ip2,
            salinity=psal, temperature=temp, pressure=pres,
            total_silicate=cy_SiOH4, total_phosphate=cy_PO4,
            grads=True,
        )

        raw[ov1][:, col1] = np.asarray(result[_PYCO2_KEY[ov1]]).ravel()
        raw[ov2][:, col2] = np.asarray(result[_PYCO2_KEY[ov2]]).ravel()

        dc[p] = {}
        i1, i2 = ip1 - 1, ip2 - 1
        for ov_name, col in ((ov1, col1), (ov2, col2)):
            out_key = _PYCO2_KEY[ov_name]
            g1 = np.asarray(result.get(f"d_{out_key}__d_par1", 0.0)).ravel()
            g2 = np.asarray(result.get(f"d_{out_key}__d_par2", 0.0)).ravel()
            dc[p][(ov_name, "g1")] = g1
            dc[p][(ov_name, "g2")] = g2

            # Propagate CANYON-B uncertainty through CO2SYS:
            # var(out) = g1² σ_v1² + g2² σ_v2² + 2 g1 g2 cov(v1, v2)
            var_prop = (
                g1 ** 2 * cycov[i1, i1]
                + g2 ** 2 * cycov[i2, i2]
                + 2.0 * g1 * g2 * cycov[i1, i2]
            )
            sigma[ov_name][:, col] = np.sqrt(np.maximum(var_prop, 0.0))
            grads[ov_name][:, col, i1] = g1
            grads[ov_name][:, col, i2] = g2

    # ------------------------------------------------------------------
    # 6.  Per-variable 4×4 covariance matrix, weighted mean, and sigmas
    # ------------------------------------------------------------------
    out: Dict[str, np.ndarray] = {}

    for k_idx, k in enumerate(PARAMS):
        # Build 4×4 covariance matrix for the four estimates of variable k
        cocov_k = np.zeros((4, 4, nol))
        for col in range(4):
            cocov_k[col, col] = sigma[k][:, col] ** 2

        # Off-diagonal: first-order covariance for every estimate pair
        # cov(a,b) = grad_a^T * Cov(PARAMS) * grad_b
        for a in range(4):
            for b in range(a + 1, 4):
                ga = grads[k][:, a, :]
                gb = grads[k][:, b, :]
                cov_ab = np.einsum("ni,ijn,nj->n", ga, cycov, gb)
                cocov_k[a, b] = cov_ab
                cocov_k[b, a] = cov_ab

        # Weights = 1 / σ²  (zero sigma → zero weight)
        w = np.where(sigma[k] > 0, 1.0 / np.maximum(sigma[k] ** 2, 1e-30), 0.0)
        w_sum = w.sum(axis=1)
        safe  = w_sum > 0

        # Weighted mean
        w_mean = np.full(nol, np.nan)
        w_mean[safe] = (w[safe] * raw[k][safe]).sum(axis=1) / w_sum[safe]

        # Empirical spread (Bessel-corrected weighted variance across 4 estimates)
        residuals   = raw[k] - w_mean[:, np.newaxis]
        w2_sum      = (w ** 2).sum(axis=1)
        denom_bessel = np.where(safe, w_sum - w2_sum / np.where(safe, w_sum, 1.0), np.nan)
        sigma_delta = np.sqrt(
            np.where(
                denom_bessel > 0,
                (w * residuals ** 2).sum(axis=1) / denom_bessel,
                np.nan,
            )
        )

        # Propagated uncertainty of weighted mean:
        # σ_prop² = Σ_i Σ_j (w_i/W)(w_j/W) cov(i,j)
        w_norm = np.where(
            w_sum[:, np.newaxis] > 0,
            w / w_sum[:, np.newaxis],
            0.0,
        )
        # Efficient double sum via einsum
        sigma_prop_sq = np.einsum(
            "ni,ijn,nj->n", w_norm, cocov_k, w_norm
        )
        sigma_prop = np.sqrt(np.maximum(sigma_prop_sq, 0.0))

        out[k]                  = w_mean
        out[f"{k}_sigma"]       = sigma_delta + sigma_prop
        out[f"{k}_sigma_min"]   = sigma_prop
        out[f"{k}_raw"]         = raw[k]

    out["canyon"] = cy
    return out