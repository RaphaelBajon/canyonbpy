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
from typing import Dict, List, Optional, Union

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

        The output dataset shares the same dimensions and coordinates as the
        source dataset, so it can be merged back trivially:

        >>> results = ds.canyonb.predict(param=["pH"])
        >>> xr.merge([ds, results])

        Parameters
        ----------
        var_map : dict, optional
            Mapping ``{canyonb_arg: dataset_variable_name}``.  Only supply
            keys that differ from the defaults:

            .. list-table::
               :header-rows: 1

               * - canyonb argument
                 - Default variable name
               * - ``gtime``
                 - ``time``
               * - ``lat``
                 - ``latitude``
               * - ``lon``
                 - ``longitude``
               * - ``pres``
                 - ``pressure``
               * - ``temp``
                 - ``temperature``
               * - ``psal``
                 - ``salinity``
               * - ``doxy``
                 - ``doxy``

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

        return self._pack_results(raw, original_shape, var_map=var_map)

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
        var_map: Optional[Dict[str, str]] = None,
    ) -> xr.Dataset:
        """Pack raw numpy outputs into an xr.Dataset with proper coordinates."""
        effective_map = {**_DEFAULT_VAR_MAP, **(var_map or {})}
        ref_var_name = effective_map["pres"]

        try:
            ref_da = (
                self._obj[ref_var_name]
                if ref_var_name in self._obj
                else self._obj.coords[ref_var_name]
            )
            dims = ref_da.dims
            coords = {d: self._obj.coords[d] for d in dims if d in self._obj.coords}
        except Exception:
            dims = ("points",)
            coords = {}

        data_vars = {}
        for key, arr in raw.items():
            arr = np.asarray(arr)
            if arr.shape == original_shape:
                # Already the right shape (most outputs)
                data_vars[key] = xr.DataArray(arr, dims=dims, coords=coords)
            elif arr.size == 1:
                # Scalar uncertainty (e.g. _cim for carbonate params where
                # cvalcimeas = inputsigma[i]**2 is a constant) — broadcast
                data_vars[key] = xr.DataArray(
                    np.full(original_shape, float(arr)), dims=dims, coords=coords
                )
            else:
                # General case: try reshaping, fall back to broadcasting
                try:
                    data_vars[key] = xr.DataArray(
                        arr.reshape(original_shape), dims=dims, coords=coords
                    )
                except ValueError:
                    data_vars[key] = xr.DataArray(
                        np.broadcast_to(arr, original_shape).copy(), dims=dims, coords=coords
                    )

        return xr.Dataset(data_vars)
