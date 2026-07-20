# Basic Usage

A single-point prediction using plain `numpy` arrays. `canyonb` returns
plain `np.ndarray` values (not `xr.DataArray`), so results are indexed
directly rather than via `.values`.

```python
"""
Basic example of using CANYON-B for single point prediction.
"""

import numpy as np
from datetime import datetime
from canyonbpy import canyonb

def main():
    # Example data for a single point
    data = {
        'gtime': [datetime(2024, 1, 1)],
        'lat': np.array([45.0]),          # North Atlantic
        'lon': np.array([-20.0]),         # North Atlantic
        'pres': np.array([100.0]),        # 100 dbar
        'temp': np.array([15.0]),         # 15°C
        'psal': np.array([35.0]),         # Salinity 35
        'doxy': np.array([250.0])         # 250 µmol/kg oxygen
    }

    # Make predictions
    results = canyonb(
        **data,
        param=['pH', 'AT'],  # Only predict pH and Alkalinity
    )

    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    for param in ['pH', 'AT']:
        value = results[param][0]
        uncertainty = results[f'{param}_ci'][0]
        print(f"{param}: {value:.3f} ± {uncertainty:.3f}")

if __name__ == "__main__":
    main()
```