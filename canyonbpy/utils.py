import numpy as np
from typing import Dict, Union
import xarray as xr
from datetime import datetime
from matplotlib.path import Path
from importlib import resources


def calculate_decimal_year(gtime: np.ndarray) -> np.ndarray:
    """
    Convert datetime to decimal year. If gtime is already a float, return it as it is, assuming it is in decimal year. 

    Args:
        gtime: Time to convert to decimal year
        
    Returns:
        np.ndarray: Time converted in decimal year
    """
    if isinstance(gtime[0], (float, np.float64)):
        # Assuming input is already in decimal years
        return gtime
    
    years = np.array([d.year + (d.dayofyear - 1 + d.hour/24 + d.minute/1440) / 366 
                      if hasattr(d, 'dayofyear') 
                      else d.year + (d - datetime(d.year, 1, 1)).total_seconds()/(366*86400)
                      for d in gtime])
    return years

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
    # Implementation
    if weights_dir is not None:
        # If custom directory provided, use it directly
        filepath = f"{weights_dir}wgts_{param_name}.txt"
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
