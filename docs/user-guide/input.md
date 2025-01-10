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

    All input arrays must:

    * Have the same shape
    * Be numpy arrays or convertible to numpy arrays
    * Can contain non-finite values (`np.nan`), but not recommanded for coordinates (`time`, `longitude`, `latitude`, `pressure`)

```python
import numpy as np
from datetime import datetime

# Single point
data = {
    'gtime': [datetime(2024, 1, 1)],
    'lat': np.array([45.0]),
    'lon': np.array([-20.0]),
    'pres': np.array([100.0]),
    'temp': np.array([15.0]),
    'psal': np.array([35.0]),
    'doxy': np.array([250.0])
}

# Multiple points/profile
depths = np.array([0, 100, 200, 500])
data_profile = {
    'gtime': [datetime(2024, 1, 1)] * len(depths),
    'lat': np.full_like(depths, 45.0),
    'lon': np.full_like(depths, -20.0),
    'pres': depths,
    'temp': np.array([20.0, 15.0, 12.0, 8.0]),
    'psal': np.full_like(depths, 35.0),
    'doxy': np.array([250.0, 220.0, 200.0, 180.0])
}
```

### Time Format

CANYON-B accepts two time formats:

1. Python datetime objects:
```python
from datetime import datetime
time = [datetime(2024, 1, 1)]
```

2. Decimal years:
```python
time = [2024.0]  # January 1st, 2024
```

### Geographic Coordinates

!!! inputs "Latitude"

    * Must be between -90° and 90°
    * Negative values for Southern hemisphere and positive values for Northern hemisphere

!!! inputs "Longitude"

    Can be in either (Automatically converted internally):

    * -180° to 180° format (preferred)
    * 0° to 360° format


!!! success "Arctic Region Handling"

    For measurements in the Arctic basin west of the Lomonosov ridge:
    
    * Latitude values are automatically adjusted
    * Adjustments account for water mass distribution
    * No user action required
