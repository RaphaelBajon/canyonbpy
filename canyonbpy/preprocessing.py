"""
Preprocessing utilities for canyonbpy.

Provides helpers to extract numpy arrays from xarray Datasets
so that ``canyonb`` / ``canyonb_from_dataset`` can work directly
on ocean model output or Argo float data stored as ``xr.Dataset``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple


# Default variable-name mapping: canyonb argument → common CF / Argo names
_DEFAULT_VAR_MAP: Dict[str, str] = {
    "gtime": "time",
    "lat":   "latitude",
    "lon":   "longitude",
    "pres":  "pressure",
    "temp":  "temperature",
    "psal":  "salinity",
    "doxy":  "doxy",
}


class DatasetToNumpy:
    """Extract and flatten numpy arrays from an :class:`xarray.Dataset`.

    This class maps ``canyonb`` input argument names to variables (or
    coordinates) in an ``xr.Dataset``, returning flat numpy arrays that
    can be passed directly to :func:`canyonbpy.canyonb`.

    Parameters
    ----------
    dataset : xr.Dataset
        The source dataset.  Must contain variables (or coordinates) for
        every field listed in *var_map*.
    var_map : dict, optional
        Mapping from ``canyonb`` keyword names to variable names in
        *dataset*.  Defaults to :data:`_DEFAULT_VAR_MAP`.  You only
        need to supply entries that differ from the defaults.

    Examples
    --------
    Basic usage with default variable names:

    >>> converter = DatasetToNumpy(ds)
    >>> numpy_inputs = converter.to_dict()
    >>> results = canyonb(**numpy_inputs)

    With a custom variable mapping (e.g. Argo BGC delayed-mode):

    >>> var_map = {
    ...     "temp": "TEMP_ADJUSTED",
    ...     "psal": "PSAL_ADJUSTED",
    ...     "doxy": "DOXY_ADJUSTED",
    ...     "pres": "PRES_ADJUSTED",
    ... }
    >>> converter = DatasetToNumpy(ds, var_map=var_map)
    >>> results = canyonb(**converter.to_dict())
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        var_map: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(
                f"Expected an xarray.Dataset, got {type(dataset).__name__}."
            )
        self.dataset = dataset
        # Merge user-supplied map on top of defaults
        self._var_map: Dict[str, str] = {**_DEFAULT_VAR_MAP, **(var_map or {})}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of flat numpy arrays ready for :func:`canyonb`.

        All input variables are broadcast to their common shape before
        flattening.  This means a 1-D pressure coordinate is automatically
        tiled over a ``(time, pressure)`` grid, for example.

        Returns
        -------
        dict
            Keys are ``canyonb`` argument names; values are 1-D numpy arrays.

        Raises
        ------
        KeyError
            If a required variable is absent from the dataset.
        """
        broadcast = self._broadcast_all()
        out: Dict[str, np.ndarray] = {}
        for canyon_key, da in broadcast.items():
            if canyon_key == "gtime":
                out[canyon_key] = self._convert_time(da)
            else:
                out[canyon_key] = da.values.flatten()
        return out

    def original_shape(self) -> Tuple[int, ...]:
        """Return the common broadcast shape of all input fields.

        Useful to reshape canyonb outputs back to the original grid.

        Returns
        -------
        tuple of int
        """
        broadcast = self._broadcast_all()
        return next(iter(broadcast.values())).shape

    def original_dims(self) -> Tuple[str, ...]:
        """Return the dimension names of the common broadcast shape.

        Returns
        -------
        tuple of str
        """
        broadcast = self._broadcast_all()
        return next(iter(broadcast.values())).dims

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _broadcast_all(self) -> Dict[str, xr.DataArray]:
        """Broadcast every input variable to the common grid shape.

        Multi-dimensional arrays are passed to ``xr.broadcast`` first so their
        dimension order is used as the canonical order for the result.
        """
        das = {k: self._get_variable(v) for k, v in self._var_map.items()}
        # Sort by ndim descending so that the variable with the most dimensions
        # anchors the broadcast dimension order.
        ordered = sorted(das.items(), key=lambda kv: -kv[1].ndim)
        broadcast_vals = xr.broadcast(*[da for _, da in ordered])
        broadcast_map = dict(zip([k for k, _ in ordered], broadcast_vals))
        return {k: broadcast_map[k] for k in das}

    def _get_variable(self, name: str) -> xr.DataArray:
        if name in self.dataset:
            return self.dataset[name]
        if name in self.dataset.coords:
            return self.dataset.coords[name]
        raise KeyError(
            f"Variable '{name}' not found in dataset.  "
            f"Available variables: {list(self.dataset.data_vars)}; "
            f"coordinates: {list(self.dataset.coords)}."
        )

    def _extract(self, ds_name: str, canyon_key: str) -> np.ndarray:  # kept for back-compat
        da = self._get_variable(ds_name)
        if canyon_key == "gtime":
            return self._convert_time(da)
        return da.values.flatten()

    @staticmethod
    def _convert_time(da: xr.DataArray) -> np.ndarray:
        """Convert a time DataArray to an array of :class:`datetime.datetime`."""
        import pandas as pd
        values = da.values.flatten()
        timestamps = pd.to_datetime(values)
        return np.array([ts.to_pydatetime() for ts in timestamps])
