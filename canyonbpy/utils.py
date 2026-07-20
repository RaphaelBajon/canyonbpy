import calendar
import os
import numpy as np
import pandas as pd
from typing import Dict, Union
import xarray as xr
from matplotlib.path import Path
from importlib import resources


def calculate_decimal_year(gtime: np.ndarray) -> np.ndarray:
    """
    Convert datetime-like values to decimal year.

    Accepts Python ``datetime`` objects, ``numpy.datetime64`` /
    pandas ``Timestamp`` values, date strings, or an array already in
    decimal-year form (in which case it is returned unchanged, cast to
    float).

    Each year is divided by its true length (365 or 366 days) so that,
    unlike a fixed 366-day divisor, the fractional part correctly reaches
    1.0 at the end of both leap and non-leap years.

    Args:
        gtime: Time to convert to decimal year

    Returns:
        np.ndarray: Time converted to decimal year, same shape as ``gtime``
    """
    arr = np.asarray(gtime)

    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        # Assuming input is already in decimal years
        return arr.astype(float)

    shape = arr.shape
    timestamps = pd.to_datetime(arr.ravel())

    year = timestamps.year.to_numpy()
    days_in_year = np.where([calendar.isleap(y) for y in year], 366, 365)
    seconds_into_year = (
        (timestamps.dayofyear.to_numpy() - 1) * 86400
        + timestamps.hour.to_numpy() * 3600
        + timestamps.minute.to_numpy() * 60
        + timestamps.second.to_numpy()
        + timestamps.microsecond.to_numpy() / 1e6
    )

    decimal_year = year + seconds_into_year / (days_in_year * 86400)
    return decimal_year.reshape(shape)

def adjust_arctic_latitude(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Adjust latitude for Arctic basin calculations.

    Args:
        lat: Latitudes to be adjusted
        lon: Corresponding longitudes 
        
    Returns:
        np.ndarray: Ajusted latitudes 
    """
    # Points for Arctic basin 'West' of Lomonossov ridge
    plon = np.array([-180, -170, -85, -80, -37, -37, 143, 143, 180, 180, -180, -180])
    plat = np.array([68, 66.5, 66.5, 80, 80, 90, 90, 68, 68, 90, 90, 68])
    
    # Convert longitude to -180 to 180 range
    lon = np.where(lon > 180, lon - 360, lon)
    
    # Create masks for points inside the polygon
    polygon = Path(np.column_stack((plon, plat)))
    points = np.column_stack((lon.flatten(), lat.flatten()))
    mask = polygon.contains_points(points)
    mask = mask.reshape(lat.shape)
    
    # Adjust latitude
    adjusted_lat = lat.copy()
    adjusted_lat[mask] = lat[mask] - np.sin(np.radians(lon[mask] + 37)) * (90 - lat[mask]) * 0.5
    
    return adjusted_lat

def load_weight_file(weights_dir: Union[str, None], param_name: str) -> np.ndarray:
    """
    Load neural network weights from file.

    Args:
        weights_dir: Optional custom directory path for weights. If None, uses package's data directory
        param_name: Name of the parameter whose weights to load

    Returns:
        np.ndarray: Loaded weights
    """
    if weights_dir is not None:
        # If custom directory provided, use it directly (with or without trailing separator)
        filepath = os.path.join(weights_dir, f"wgts_{param_name}.txt")
        return np.loadtxt(filepath)
    # If no custom directory, use package data
    with resources.open_text('canyonbpy.data.weights', f'wgts_{param_name}.txt') as f:
        return np.loadtxt(f)

def create_output_array(data: np.ndarray, coords: Dict) -> xr.DataArray:
    """Create xarray DataArray with proper coordinates."""
    # Implementation
    pass

def calculate_pco2_corrections(ct_pred: np.ndarray, temp: np.ndarray, 
                             psal: np.ndarray) -> np.ndarray:
    """Calculate pCO2 corrections using PyCO2SYS."""
    # Implementation
    pass
