# Core API Reference

## Main Function

### `canyonb`

```python
def canyonb(
    gtime: Union[np.ndarray, List],
    lat: np.ndarray,
    lon: np.ndarray,
    pres: np.ndarray,
    temp: np.ndarray,
    psal: np.ndarray,
    doxy: np.ndarray,
    param: Optional[List[str]] = None,
    epres: Optional[float] = 0.5,
    etemp: Optional[float] = 0.005,
    epsal: Optional[float] = 0.005,
    edoxy: Optional[Union[float, np.ndarray]] = None,
    weights_dir: str = None
) -> Dict[str, xr.DataArray]
```

Primary function for neural network predictions of ocean parameters.

#### Parameters

##### Required Inputs
- `gtime`: Date/time of measurements
    - Type: datetime objects or decimal years
    - Shape: Must match other inputs
- `lat`: Latitude
    - Type: numpy array
    - Range: -90 to 90
- `lon`: Longitude
    - Type: numpy array
    - Range: -180 to 180 or 0 to 360
- `pres`: Pressure
    - Type: numpy array
    - Units: dbar
- `temp`: In-situ temperature
    - Type: numpy array
    - Units: °C
- `psal`: Salinity
    - Type: numpy array
- `doxy`: Dissolved oxygen
    - Type: numpy array
    - Units: µmol/kg

##### Optional Parameters
- `param`: List of parameters to calculate
    - Type: list of strings
    - Default: All parameters
    - Options: ['AT', 'CT', 'pH', 'pCO2', 'NO3', 'PO4', 'SiOH4']
- `epres`: Pressure measurement error
    - Type: float
    - Default: 0.5
- `etemp`: Temperature measurement error
    - Type: float
    - Default: 0.005
- `epsal`: Salinity measurement error
    - Type: float
    - Default: 0.005
- `edoxy`: Oxygen measurement error
    - Type: float or array
    - Default: 1% of doxy value
- `weights_dir`: Custom weights directory
    - Type: string
    - Default: None (uses package weights)

#### Returns

Dictionary containing predictions and uncertainties:

```python
{
    'parameter': np.ndarray,          # Main prediction
    'parameter_ci': np.ndarray,       # Total uncertainty
    'parameter_cim': np.ndarray,      # Measurement uncertainty
    'parameter_cin': np.ndarray,      # Neural network uncertainty
    'parameter_cii': np.ndarray       # Input uncertainty
}
```

#### Examples

```python
# Basic usage
results = canyonb(**data)

# Specific parameters
results = canyonb(**data, param=['pH', 'AT'])

# Custom errors
results = canyonb(**data, epres=1.0, etemp=0.01)
```

#### Notes

1. Input arrays must all have the same shape
2. NaN values in inputs will result in NaN predictions
3. Arctic latitudes are automatically adjusted
4. Time can be provided as datetime objects or decimal years