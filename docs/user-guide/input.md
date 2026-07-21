# Input Data

## Required inputs

The function `canyonb` requires seven input parameters for prediction:

| Parameter | Description | Units | Range | Typical Error |
|-----------|-------------|-------|-------|---------------|
| Time | Measurement date | UTC or decimal year | Any | - |
| Latitude | Geographic latitude | degrees | -90 to 90 | - |
| Longitude | Geographic longitude | degrees | -180 to 180 | - |
| Pressure | Water pressure | dbar | 0 to 10000 | 0.5 |
| Temperature | In-situ temperature | °C | -2 to 35 | 0.005 |
| Salinity | Practical salinity | PSU | 0 to 45 | 0.005 |
| Oxygen | Dissolved oxygen | μmol·kg<sup>−1</sup> | 0 to 400 | 1% of value |

## Data Format

!!! inputs "Array Requirements"

    All input arrays:

    * Must be numpy arrays or convertible to numpy arrays
    * Can contain non-finite values (`np.nan`), but not recommanded for coordinates (`time`, `longitude`, `latitude`, `pressure`)

### Time Format

CANYON-B accepts any of the following time formats (converted internally by
`calculate_decimal_year`):

1. Python `datetime` objects:
```python
from datetime import datetime
time = [datetime(2024, 1, 1)]
```

2. `numpy.datetime64` / pandas `Timestamp` arrays — e.g. directly from an
   `xarray` time coordinate:
```python
time = ds["time"].values  # dtype='datetime64[ns]'
```

3. ISO date/time strings:
```python
time = ["2024-01-01", "2024-06-30T12:00:00"]
```

4. Decimal years:
```python
time = [2024.0]  # January 1st, 2024
```

!!! note "Leap years"

    The fractional part of the decimal year is computed against the true
    length of that calendar year (365 or 366 days), so `2024-12-31 23:59:59`
    (a leap year) and `2023-12-31 23:59:59` (not a leap year) both resolve
    to just under the following year, rather than being off by up to a day.

### Geographic Coordinates

!!! inputs "Longitude"

    Can be in either (Automatically converted internally):

    * -180° to 180° format (preferred)
    * 0° to 360° format
