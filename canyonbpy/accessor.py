"""
xarray accessor for canyonbpy.

Registers a ``canyonb`` accessor on :class:`xarray.Dataset` objects so that
CANYON-B predictions can be run directly from a dataset without any manual
variable extraction::

    import canyonbpy  # accessor is registered on import
    results = ds.canyonb.predict(param=["pH", "NO3"])

The accessor is registered under the name ``"canyonb"``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union

from .preprocessing import DatasetToNumpy, _DEFAULT_VAR_MAP


@xr.register_dataset_accessor("canyonb")
class CanyonBAccessor:
    """xarray accessor to run CANYON-B predictions on a :class:`xarray.Dataset`.

    Registered under ``ds.canyonb`` when ``canyonbpy`` is imported.

    Parameters
    ----------
    xarray_obj : xr.Dataset
        The dataset to which this accessor is attached.

    Examples
    --------
    With default variable names (``time``, ``latitude``, ``longitude``,
    ``pressure``, ``temperature``, ``salinity``, ``doxy``):

    >>> import canyonbpy
    >>> results = ds.canyonb.predict()
    >>> ds_enriched = xr.merge([ds, results])

    Run CONTENT for internally-consistent carbonate estimates:

    >>> results = ds.canyonb.content()
    >>> ds_enriched = xr.merge([ds, results])

    Predict only a subset of parameters:

    >>> results = ds.canyonb.predict(param=["pH", "AT", "NO3"])

    Use a custom variable mapping (e.g. Argo BGC delayed-mode):

    >>> var_map = {
    ...     "temp": "TEMP_ADJUSTED",
    ...     "psal": "PSAL_ADJUSTED",
    ...     "doxy": "DOXY_ADJUSTED",
    ...     "pres": "PRES_ADJUSTED",
    ...     "lat":  "LATITUDE",
    ...     "lon":  "LONGITUDE",
    ... }
    >>> results = ds.canyonb.predict(var_map=var_map)

    Access the underlying :class:`~canyonbpy.preprocessing.DatasetToNumpy`
    converter directly:

    >>> converter = ds.canyonb.converter()
    >>> inputs = converter.to_dict()   # dict[str, np.ndarray]
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        if not isinstance(xarray_obj, xr.Dataset):
            raise TypeError(
                f"The canyonb accessor is only available on xr.Dataset objects, "
                f"got {type(xarray_obj).__name__}."
            )
        self._obj = xarray_obj

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def predict(
        self,
        var_map: Optional[Dict[str, str]] = None,
        param: Optional[List[str]] = None,
        epres: float = 0.5,
        etemp: float = 0.005,
        epsal: float = 0.005,
        edoxy: Optional[Union[float, np.ndarray]] = None,
        weights_dir: Optional[str] = None,
    ) -> xr.Dataset:
        """Run CANYON-B and return predictions as an :class:`xarray.Dataset`.

        Parameters
        ----------
        var_map : dict, optional
            Mapping ``{canyonb_arg: dataset_variable_name}``.  Only supply
            keys that differ from the defaults:

            | canyonb argument | Default variable name |
            |------------------|-----------------------|
            | ``gtime``        | ``time``              |
            | ``lat``          | ``latitude``          |
            | ``lon``          | ``longitude``         |
            | ``pres``         | ``pressure``          |
            | ``temp``         | ``temperature``       |
            | ``psal``         | ``salinity``          |
            | ``doxy``         | ``doxy``              |

        param : list of str, optional
            Parameters to compute.  Defaults to all:
            ``['AT', 'CT', 'pH', 'pCO2', 'NO3', 'PO4', 'SiOH4']``.
        epres, etemp, epsal : float, optional
            Measurement errors for pressure, temperature and salinity.
        edoxy : float or array-like, optional
            Oxygen measurement error.  Defaults to 1 % of ``doxy``.
        weights_dir : str, optional
            Path to a directory containing custom weight files.

        Returns
        -------
        xr.Dataset
            Predictions and uncertainties as ``xr.DataArray`` variables,
            sharing dimensions and coordinates with the source dataset.
        """
        from .core import canyonb

        conv = self.converter(var_map=var_map)
        original_shape = conv.original_shape()
        original_dims  = conv.original_dims()
        numpy_inputs = conv.to_dict()

        raw = canyonb(
            **numpy_inputs,
            param=param,
            epres=epres,
            etemp=etemp,
            epsal=epsal,
            edoxy=edoxy,
            weights_dir=weights_dir,
        )

        return self._pack_results(raw, original_shape, original_dims, var_map=var_map)

    def content(
        self,
        var_map: Optional[Dict[str, str]] = None,
        epres: float = 0.5,
        etemp: float = 0.005,
        epsal: float = 0.005,
        edoxy: Optional[Union[float, np.ndarray]] = None,
        include_raw: bool = False,
        include_canyonb: bool = False,
    ) -> xr.Dataset:
        """Run CONTENT and return internally-consistent carbonate estimates.

        Combines CANYON-B neural-network estimates of AT, CT, pH, and pCO₂
        with all six two-parameter CO2SYS combinations into a weighted-mean
        estimate with full propagated uncertainty.

        Parameters
        ----------
        var_map : dict, optional
            Custom variable name mapping (same semantics as in :meth:`predict`).
        epres, etemp, epsal : float, optional
            Measurement errors for pressure, temperature and salinity.
        edoxy : float or array-like, optional
            Oxygen measurement error.  Defaults to 1 % of ``doxy``.
        include_raw : bool, optional
            If ``True``, add ``AT_raw``, ``CT_raw``, ``pH_raw``, ``pCO2_raw``
            to the output dataset as ``(N, 4)`` arrays along an ``estimate``
            dimension (col 0 = CANYON-B direct, cols 1–3 = CO2SYS indirect).
            Default ``False``.
        include_canyonb : bool, optional
            If ``True``, include the full CANYON-B sub-output (AT, CT, pH,
            pCO2, SiOH4, PO4 plus all ``_ci``/``_cii``/``_cin``/``_cim``
            components) prefixed with ``canyon_``.  Default ``False``.

        Returns
        -------
        xr.Dataset
            Variables ``AT``, ``CT``, ``pH``, ``pCO2`` (weighted-mean
            estimates) and ``*_sigma``, ``*_sigma_min`` (uncertainties),
            sharing dimensions and coordinates with the source dataset.

        Examples
        --------
        >>> import canyonbpy
        >>> results = ds.canyonb.content()
        >>> ds_enriched = xr.merge([ds, results])

        >>> # Include the 4 individual estimates per variable:
        >>> results = ds.canyonb.content(include_raw=True)
        """
        from .content import co2content

        conv = self.converter(var_map=var_map)
        original_shape = conv.original_shape()
        original_dims  = conv.original_dims()
        numpy_inputs = conv.to_dict()

        raw = co2content(
            **numpy_inputs,
            epres=epres,
            etemp=etemp,
            epsal=epsal,
            edoxy=edoxy,
        )

        return self._pack_content_results(
            raw, original_shape, original_dims,
            var_map=var_map,
            include_raw=include_raw,
            include_canyonb=include_canyonb,
        )

    def converter(
        self, var_map: Optional[Dict[str, str]] = None
    ) -> DatasetToNumpy:
        """Return a :class:`~canyonbpy.preprocessing.DatasetToNumpy` for this dataset.

        Useful when you need to inspect or modify the numpy arrays before
        calling :func:`~canyonbpy.canyonb` manually.

        Parameters
        ----------
        var_map : dict, optional
            Custom variable name mapping (same semantics as in :meth:`predict`).

        Returns
        -------
        DatasetToNumpy
        """
        return DatasetToNumpy(self._obj, var_map=var_map)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pack_results(
        self,
        raw: Dict[str, np.ndarray],
        original_shape,
        original_dims: Tuple[str, ...],
        var_map: Optional[Dict[str, str]] = None,
    ) -> xr.Dataset:
        """Pack raw numpy outputs into an xr.Dataset with proper coordinates."""
        dims = original_dims
        coords = {
            name: da
            for name, da in self._obj.coords.items()
            if all(d in dims for d in da.dims)
        }

        data_vars = {}
        for key, arr in raw.items():
            arr = np.asarray(arr)
            if arr.shape == original_shape:
                data_vars[key] = xr.DataArray(arr, dims=dims, coords=coords)
            elif arr.size == 1:
                # Scalar uncertainty (e.g. _cim for carbonate params) — broadcast
                data_vars[key] = xr.DataArray(
                    np.full(original_shape, float(arr)), dims=dims, coords=coords
                )
            else:
                try:
                    data_vars[key] = xr.DataArray(
                        arr.reshape(original_shape), dims=dims, coords=coords
                    )
                except ValueError:
                    data_vars[key] = xr.DataArray(
                        np.broadcast_to(arr, original_shape).copy(), dims=dims, coords=coords
                    )

        return xr.Dataset(data_vars)

    def _pack_content_results(
        self,
        raw: Dict,
        original_shape,
        original_dims: Tuple[str, ...],
        var_map: Optional[Dict[str, str]] = None,
        include_raw: bool = False,
        include_canyonb: bool = False,
    ) -> xr.Dataset:
        """Pack co2content numpy outputs into an xr.Dataset."""
        dims = original_dims
        coords = {
            name: da
            for name, da in self._obj.coords.items()
            if all(d in dims for d in da.dims)
        }

        _CONTENT_VARS = ["AT", "CT", "pH", "pCO2"]
        _SCALAR_SUFFIXES = ("", "_sigma", "_sigma_min")
        data_vars: Dict = {}

        # Main scalar outputs — shape (N,) → original_shape
        for v in _CONTENT_VARS:
            for suffix in _SCALAR_SUFFIXES:
                key = f"{v}{suffix}"
                arr = np.asarray(raw[key]).reshape(original_shape)
                data_vars[key] = xr.DataArray(arr, dims=dims, coords=coords)

        # Optional: 4-estimate arrays — shape (N, 4) → original_shape + (4,)
        if include_raw:
            est_dim = "estimate"
            est_coords = {**coords, est_dim: ["canyon_b", "co2sys_1", "co2sys_2", "co2sys_3"]}
            raw_dims = dims + (est_dim,)
            for v in _CONTENT_VARS:
                arr = np.asarray(raw[f"{v}_raw"]).reshape(original_shape + (4,))
                data_vars[f"{v}_raw"] = xr.DataArray(arr, dims=raw_dims, coords=est_coords)

        # Optional: full CANYON-B sub-output — prefixed with "canyonb_"
        if include_canyonb:
            nol = int(np.prod(original_shape))
            for key, val in raw["canyon"].items():
                arr = np.asarray(val)
                if arr.ndim == 1 and arr.size == nol:
                    data_vars[f"canyonb_{key}"] = xr.DataArray(
                        arr.reshape(original_shape), dims=dims, coords=coords
                    )

        return xr.Dataset(data_vars)
